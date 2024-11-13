from __future__ import print_function
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import logging
from pysmatch import *
import pysmatch.functions as uf
from catboost import CatBoostClassifier
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
class Matcher:
    """
    Matcher Class -- Match data for an observational study.

    Parameters
    ----------
    test : pd.DataFrame
        Data representing the test group
    control : (pd.DataFrame)
        Data representing the control group
    formula : str (optional)
        custom formula to use for logistic regression
        i.e. "Y ~ x1 + x2 + ..."
    yvar : str (optional)
        Name of dependent variable (the treatment)
    exclude : list  (optional)
        List of variables to ignore in regression/matching.
        Useful for unique idenifiers
    """

    def __init__(self, test, control, yvar, formula=None, exclude=None):
        if exclude is None:
            exclude = []
        plt.rcParams["figure.figsize"] = (10, 5)
        aux_match = ['scores', 'match_id', 'weight', 'record_id']
        t = test.copy().reset_index(drop=True)
        c = control.copy().reset_index(drop=True)
        t = t.dropna(axis=1, how="all")
        c = c.dropna(axis=1, how="all")
        c.index += len(t)
        self.data = pd.concat([t, c], ignore_index=True)
        self.control_color = "#1F77B4"
        self.test_color = "#FF7F0E"
        self.yvar = yvar
        self.exclude = exclude + [self.yvar] + aux_match
        self.formula = formula
        self.nmodels = 1
        self.models = []
        self.swdata = None
        self.model_accuracy = []
        self.errors = 0
        self.data[yvar] = self.data[yvar].astype(int)  # should be binary 0, 1
        self.xvars = [i for i in self.data.columns if i not in self.exclude]
        self.original_xvars = self.xvars.copy()
        self.data = self.data.dropna(subset=self.xvars)
        self.matched_data = []
        self.xvars_escaped = [f"Q('{x}')" for x in self.xvars]
        self.yvar_escaped = f"Q('{self.yvar}')"
        self.y, self.X = patsy.dmatrices(
            f'{self.yvar_escaped} ~ {" + ".join(self.xvars_escaped)}',
            data=self.data, return_type='dataframe')
        self.design_info = self.X.design_info
        self.test = self.data[self.data[yvar] == 1]
        self.control = self.data[self.data[yvar] == 0]
        self.testn = len(self.test)
        self.controln = len(self.control)
        if self.testn <= self.controln:
            self.minority, self.majority = 1, 0
        else:
            self.minority, self.majority = 0, 1
        logging.info(f'Formula:{yvar} ~ {"+".join(self.xvars)}')
        logging.info(f'n majority:{len(self.data[self.data[yvar] == self.majority])}')
        logging.info(f'n minority:{len(self.data[self.data[yvar] == self.minority])}')
    def preprocess_data(self, X, fit_scaler=False, index=None):
        X_encoded = pd.get_dummies(X)

        if not hasattr(self, 'X_columns'):
            self.X_columns = X_encoded.columns
        else:
            X_encoded = X_encoded.reindex(columns=self.X_columns, fill_value=0)

        if fit_scaler:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            if not hasattr(self, 'scalers'):
                self.scalers = {}
            self.scalers[index] = scaler
        else:
            scaler = self.scalers[index]
            X_scaled = scaler.transform(X_encoded)

        return X_scaled
    def fit_model(self, index, X, y, model_type, balance):
        X_train, _, y_train, _ = train_test_split(X, y, train_size=0.7, random_state=index)

        if balance:
            ros = RandomOverSampler(random_state=index)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train

        if model_type in ['linear', 'knn']:
            X_processed = self.preprocess_data(X_resampled, fit_scaler=True, index=index)
        else:
            X_processed = X_resampled

        if model_type == 'linear':
            model = LogisticRegression(max_iter=100)
            model.fit(X_processed, y_resampled.iloc[:, 0])
            accuracy = model.score(X_processed, y_resampled)
        elif model_type == 'tree':
            cat_features_indices = np.where(X_resampled.dtypes == 'object')[0]
            model = CatBoostClassifier(iterations=100, depth=6,
                                       eval_metric='AUC', l2_leaf_reg=3,
                                       cat_features=cat_features_indices,
                                       learning_rate=0.02, loss_function='Logloss',
                                       logging_level='Silent')
            model.fit(X_resampled, y_resampled.iloc[:, 0], plot=False)
            accuracy = model.score(X_resampled, y_resampled)
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_processed, y_resampled.iloc[:, 0])
            accuracy = model.score(X_processed, y_resampled)
        else:
            raise ValueError("Invalid model_type. Choose from 'linear', 'tree', or 'knn'.")
        logging.info(f"Model {index + 1}/{self.nmodels} trained. Accuracy: {accuracy:.2%}")
        return {'model': model, 'accuracy': accuracy}

    def fit_scores(self, balance=True, nmodels=None, n_jobs=1, model_type='linear'):
        self.models, self.model_accuracy = [], []
        self.model_type = model_type
        num_cores = mp.cpu_count()
        logging.info(f"This computer has: {num_cores} cores, The workers will be: {min(num_cores, n_jobs)}")

        if balance and nmodels is None:
            minor, major = [self.data[self.data[self.yvar] == i] for i in (self.minority, self.majority)]
            nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)

        nmodels = max(1, nmodels)

        if balance:
            with Pool(min(num_cores, n_jobs)) as pool:
                results = pool.starmap(self.fit_model,
                                       [(i, self.X, self.y, self.model_type, balance) for i in range(nmodels)])
            for res in results:
                self.models.append(res['model'])
                self.model_accuracy.append(res['accuracy'])
            logging.info(f"Average Accuracy:{np.mean(self.model_accuracy):.2%} ")
        else:
            result = self.fit_model(0, self.X, self.y, self.model_type, balance)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            logging.info(f"Accuracy:{round(self.model_accuracy[0] * 100,2)}%")

    def predict_scores(self):
        """
        Predict propensity scores for each observation.
        Adds a "scores" column to self.data

        Returns
        -------
        None
        """
        model_preds = []

        for idx, m in enumerate(self.models):
            if self.model_type in ['linear', 'knn']:
                X_processed = self.preprocess_data(self.X, fit_scaler=False, index=idx)
            else:
                X_processed = self.X

            if self.model_type == 'linear':
                preds = m.predict_proba(X_processed)[:, 1]
            elif self.model_type == 'tree':
                preds = m.predict(self.X, prediction_type='Probability')[:, 1]
            elif self.model_type == 'knn':
                preds = m.predict_proba(X_processed)[:, 1]
            else:
                raise ValueError("Invalid model_type. Choose from 'linear', 'tree', or 'knn'.")
            model_preds.append(preds)

        model_preds = np.array(model_preds)
        scores = np.mean(model_preds, axis=0)

        self.data['scores'] = scores

    def match(self, threshold=0.001, nmatches=1, method='min', max_rand=10, replacement=False):
        """
        Finds suitable match(es) for each record in the minority
        dataset, if one exists. Records are excluded from the final
        matched dataset if there are no suitable matches.

        self.matched_data contains the matched dataset once this
        method is called

        Parameters
        ----------
        threshold : float
            threshold for fuzzy matching
            i.e. |score_x - score_y| <= threshold
        nmatches : int
            How majority profiles should be matched
            (at most) to minority profiles
        method : str
            Strategy for when multiple majority profiles
            are suitable matches for a single minority profile
            "random" - choose randomly (fast, good for testing)
            "min" - choose the profile with the closest score
        max_rand : int
            max number of profiles to consider when using random tie-breaks
        replacement : bool
            Whether to allow replacement in the matching process

        Returns
        -------
        None
        """
        if 'scores' not in self.data.columns:  # Check if the propensity scores are already calculated
            logging.info("Propensity Scores have not been calculated. Using defaults...")
            self.fit_scores()  # Fit the propensity score models
            self.predict_scores()  # Predict propensity scores for the data

        test_scores = self.data[self.data[self.yvar] == True][['scores']]  # Get scores for the test group
        ctrl_scores = self.data[self.data[self.yvar] == False][['scores']]  # Get scores for the control group
        result, match_ids = [], []  # Initialize the result list and match ids
        used_indices = set()  # Keep track of used indices if no replacement is allowed

        for i in range(len(test_scores)):  # Iterate through each test score
            match_id = i
            score = test_scores.iloc[i]
            if method == 'random':  # If the method is random
                bool_match = abs(ctrl_scores - score) <= threshold  # Find all control scores within the threshold
                matches = ctrl_scores.loc[bool_match[bool_match.scores].index]
            elif method == 'min':  # If the method is minimum difference
                matches = abs(ctrl_scores - score).sort_values('scores').head(nmatches)  # Find the closest scores
            else:
                raise ValueError("Invalid method parameter, use ('random', 'min')")

            if len(matches) == 0:  # If no matches are found, continue to the next
                continue

            if not replacement:  # If replacement is not allowed
                matches = matches[~matches.index.isin(used_indices)]  # Exclude already used indices

            if len(matches) == 0:  # Check again if there are matches after filtering
                continue

            select = nmatches if method != 'random' else np.random.choice(range(1, max_rand + 1), 1)  # Select number of matches
            chosen = np.random.choice(matches.index, min(select, nmatches), replace=False)  # Choose the indices for matching

            if not replacement:  # If no replacement, update the used indices
                used_indices.update(chosen)

            result.extend([test_scores.index[i]] + list(chosen))  # Append the matched indices to the result
            match_ids.extend([i] * (len(chosen) + 1))  # Append the match_id for each pair

        self.matched_data = self.data.loc[result]  # Create the matched dataset
        self.matched_data['match_id'] = match_ids  # Assign match_id to each row in the matched dataset
        self.matched_data['record_id'] = self.matched_data.index  # Assign record_id to each row in the matched dataset

    def plot_scores(self):
        """
        Plots the distribution of propensity scores before matching between
        our test and control groups
        """
        assert 'scores' in self.data.columns, \
            "Propensity scores haven't been calculated, use Matcher.predict_scores()"
        sns.distplot(self.data[self.data[self.yvar] == 0].scores, label='Control')
        sns.distplot(self.data[self.data[self.yvar] == 1].scores, label='Test')
        plt.legend(loc='upper right')
        plt.xlim((0, 1))
        plt.title("Propensity Scores Before Matching")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Scores")
        plt.show()

    def prop_test(self, col):
        """
        Performs a Chi-Square test of independence on <col>
        See stats.chi2_contingency()

        Parameters
        ----------
        col : str
            Name of column on which the test should be performed

        Returns
        ______
        dict
            {'var': <col>,
             'before': <pvalue before matching>,
             'after': <pvalue after matching>}


        """
        if not uf.is_continuous(col, self.X) and col not in self.exclude:
            pval_before = round(stats.chi2_contingency(self.prep_prop_test(self.data,
                                                                           col))[1], 6)
            pval_after = round(stats.chi2_contingency(self.prep_prop_test(self.matched_data,
                                                                          col))[1], 6)
            return {'var': col, 'before': pval_before, 'after': pval_after}
        else:
            logging.info(f"{col} is a continuous variable")

    def compare_continuous(self, save=False, return_table=False,plot_result = True):
        """
        Plots the ECDFs for continuous features before and
        after matching. Each chart title contains test results
        and statistics to summarize how similar the two distributions
        are (we want them to be close after matching).

        Tests performed:
        Kolmogorov-Smirnov Goodness of fit Test (KS-test)
            This test statistic is calculated on 1000
            permuted samples of the data, generating
            an imperical p-value.  See pysmatch.functions.ks_boot()
            This is an adaptation of the ks.boot() method in
            the R "Matching" package
            https://www.rdocumentation.org/packages/Matching/versions/4.9-2/topics/ks.boot
        Chi-Square Distance:
            Similarly this distance metric is calculated on
            1000 permuted samples.
            See pysmatch.functions.grouped_permutation_test()

        Other included Stats:
        Standarized mean and median differences
        How many standard deviations away are the mean/median
        between our groups before and after matching
        i.e. abs(mean(control) - mean(test)) / std(control.union(test))

        Parameters
        ----------
        return_table : bool
            Should the function a table with tests and statistics?

        Returns
        -------
        pd.DataFrame (optional)
            Table of before/after statistics if return_table == True


        """
        test_results = []
        for col in self.matched_data.columns:
            if uf.is_continuous(col, self.X) and col not in self.exclude:
                # organize data
                trb, cob = self.test[col], self.control[col]
                tra = self.matched_data[self.matched_data[self.yvar] == True][col]
                coa = self.matched_data[self.matched_data[self.yvar] == False][col]
                xtb, xcb = ECDF(trb), ECDF(cob)
                xta, xca = ECDF(tra), ECDF(coa)

                # before/after stats
                std_diff_med_before, std_diff_mean_before = uf.std_diff(trb, cob)
                std_diff_med_after, std_diff_mean_after = uf.std_diff(tra, coa)
                pb, truthb = uf.grouped_permutation_test(uf.chi2_distance, trb, cob)
                pa, trutha = uf.grouped_permutation_test(uf.chi2_distance, tra, coa)
                ksb = round(uf.ks_boot(trb, cob, nboots=1000), 6)
                ksa = round(uf.ks_boot(tra, coa, nboots=1000), 6)
                if plot_result:
                    # plotting
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))
                    ax1.plot(xcb.x, xcb.y, label='Control', color=self.control_color)
                    ax1.plot(xtb.x, xtb.y, label='Test', color=self.test_color)
                    ax1.plot(xcb.x, xcb.y, label='Control', color=self.control_color)
                    ax1.plot(xtb.x, xtb.y, label='Test', color=self.test_color)

                    title_str = '''
                    ECDF for {} {} Matching
                    KS p-value: {}
                    Grouped Perm p-value: {}
                    Std. Median Difference: {}
                    Std. Mean Difference: {}
                    '''
                    ax1.set_title(title_str.format(col, "before", ksb, pb,
                                                   std_diff_med_before, std_diff_mean_before))
                    ax2.plot(xca.x, xca.y, label='Control')
                    ax2.plot(xta.x, xta.y, label='Test')
                    ax2.set_title(title_str.format(col, "after", ksa, pa,
                                                   std_diff_med_after, std_diff_mean_after))
                    ax2.legend(loc="lower right")
                    plt.xlim((0, np.percentile(xta.x, 99)))
                    plt.show()

                test_results.append({
                    "var": col,
                    "ks_before": ksb,
                    "ks_after": ksa,
                    "grouped_chisqr_before": pb,
                    "grouped_chisqr_after": pa,
                    "std_median_diff_before": std_diff_med_before,
                    "std_median_diff_after": std_diff_med_after,
                    "std_mean_diff_before": std_diff_mean_before,
                    "std_mean_diff_after": std_diff_mean_after
                })

        var_order = [
            "var",
            "ks_before",
            "ks_after",
            "grouped_chisqr_before",
            "grouped_chisqr_after",
            "std_median_diff_before",
            "std_median_diff_after",
            "std_mean_diff_before",
            "std_mean_diff_after"
        ]

        return pd.DataFrame(test_results)[var_order] if return_table else None

    def compare_categorical(self, return_table=False,plot_result=True):
        """
        Plots the proportional differences of each enumerated
        discete column for test and control.
        i.e. <prop_test_that_have_x>  - <prop_control_that_have_x>
        Each chart title contains the results from a
        Chi-Square Test of Independence before and after
        matching.
        See pysmatch.prop_test()

        Parameters
        ----------
        return_table : bool
            Should the function return a table with
            test results?

        Return
        ------
        pd.DataFrame() (optional)
            Table with the p-values of the Chi-Square contingency test
            for each discrete column before and after matching

        """

        def prep_plot(data, var, colname):
            t, c = data[data[self.yvar] == 1], data[data[self.yvar] == 0]
            # dummy var for counting
            dummy = [i for i in t.columns if i not in \
                     (var, "match_id", "record_id", "weight")][0]
            countt = t[[var, dummy]].groupby(var).count() / len(t)
            countc = c[[var, dummy]].groupby(var).count() / len(c)
            ret = (countt - countc).dropna()
            ret.columns = [colname]
            return ret

        title_str = '''
        Proportional Difference (test-control) for {} Before and After Matching
        Chi-Square Test for Independence p-value before | after:
        {} | {}
        '''
        test_results = []
        for col in self.matched_data.columns:
            if not uf.is_continuous(col, self.X) and col not in self.exclude:
                dbefore = prep_plot(self.data, col, colname="before")
                dafter = prep_plot(self.matched_data, col, colname="after")
                df = dbefore.join(dafter)
                test_results_i = self.prop_test(col)
                test_results.append(test_results_i)
                if plot_result:
                    # plotting
                    df.plot.bar(alpha=.8)
                    plt.title(title_str.format(col, test_results_i["before"],
                                               test_results_i["after"]))
                    lim = max(.09, abs(df).max().max()) + .01
                    plt.ylim((-lim, lim))
        return pd.DataFrame(test_results)[['var', 'before', 'after']] if return_table else None

    def prep_prop_test(self, data, var):
        """
        Helper method for running chi-square contingency tests

        Balances the counts of discrete variables with our groups
        so that missing levels are replaced with 0.
        i.e. if the test group has no records with x as a field
        for a given column, make sure the count for x is 0
        and not missing.

        Parameters
        ----------
        data : pd.DataFrame()
            Data to use for counting
        var : str
            Column to use within data

        Returns
        -------
        list
            A table (list of lists) of counts for all enumerated field within <var>
            for test and control groups.
        """
        counts = data.groupby([var, self.yvar]).count().reset_index()
        table = []
        for t in (0, 1):
            os_counts = counts[counts[self.yvar] == t] \
                .sort_values(var)
            cdict = {}
            for row in os_counts.iterrows():
                row = row[1]
                cdict[row[var]] = row[2]
            table.append(cdict)
        # fill empty keys as 0
        all_keys = set(chain.from_iterable(table))
        for d in table:
            d.update((k, 0) for k in all_keys if k not in d)
        ctable = [[i[k] for k in sorted(all_keys)] for i in table]
        return ctable

    def prop_retained(self):
        """
        Returns the proportion of data retained after matching
        """
        return len(self.matched_data[self.matched_data[self.yvar] == self.minority]) * 1.0 / \
            len(self.data[self.data[self.yvar] == self.minority])

    def tune_threshold(self, method, nmatches=1, rng=np.arange(0, .001, .0001)):
        """
        Matches data over a grid to optimize threshold value and plots results.

        Parameters
        ----------
        method : str
            Method used for matching (use "random" for this method)
        nmatches : int
            Max number of matches per record. See pysmatch.match()
        rng: : list / np.array()
            Grid of threshold values

        Returns
        -------
        None

        """
        results = []
        for i in rng:
            self.match(method=method, nmatches=nmatches, threshold=i)
            results.append(self.prop_retained())
        plt.plot(rng, results)
        plt.title("Proportion of Data retained for grid of threshold values")
        plt.ylabel("Proportion Retained")
        plt.xlabel("Threshold")
        plt.xticks(rng)
        plt.show()

    def record_frequency(self):
        """
        Calculates the frequency of specifi records in
        the matched dataset

        Returns
        -------
        pd.DataFrame()
            Frequency table of the number records
            matched once, twice, ..., etc.
        """
        freqs = self.matched_data.groupby("record_id") \
            .count().groupby("match_id").count() \
            [["scores"]].reset_index()
        freqs.columns = ["freq", "n_records"]
        return freqs

    def assign_weight_vector(self):
        record_freqs = self.matched_data.groupby('record_id').size().reset_index(name='count')
        record_freqs['weight'] = 1 / record_freqs['count']
        self.matched_data = self.matched_data.merge(record_freqs[['record_id', 'weight']], on='record_id')

    def _scores_to_accuracy(self, m, X, y):
        if self.model_type == 'linear':
            preds = m.predict(X)
        else:
            preds = m.predict(X, prediction_type='Probability', ntree_start=0, ntree_end=0, thread_count=-1,
                              verbose=None)[:, 1]

        preds = (preds >= 0.5).astype(float)

        accuracy = np.mean(preds == y.to_numpy())
        return accuracy
