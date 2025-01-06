from __future__ import print_function
import matplotlib.pyplot as plt
import logging

from sklearn.neighbors import NearestNeighbors

from pysmatch import *
import pysmatch.utils as uf
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import List, Optional
import patsy
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats

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

    def __init__(self, test: pd.DataFrame, control: pd.DataFrame, yvar: str,
                 formula: Optional[str] = None, exclude: Optional[List[str]] = None):
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


    def fit_model(self, index: int, X: pd.DataFrame, y: pd.Series, model_type: str, balance: bool, max_iter: int = 100) -> dict:
        # 动态引入与模型类型相关的库
        if model_type == 'tree':
            from catboost import CatBoostClassifier
        elif model_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
        elif model_type == 'linear':
            from sklearn.linear_model import LogisticRegression

        X_train, _, y_train, _ = train_test_split(X, y, train_size=0.7, random_state=index)

        if balance:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=index)
            X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train

        numerical_features = X_resampled.select_dtypes(include=np.number).columns
        categorical_features = X_resampled.select_dtypes(exclude=np.number).columns

        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

        # 根据模型类型初始化模型
        if model_type == 'linear':
            model = LogisticRegression(max_iter=max_iter)
        elif model_type == 'tree':
            model = CatBoostClassifier(iterations=max_iter, depth=6, eval_metric='AUC', l2_leaf_reg=3,
                                       learning_rate=0.02, loss_function='Logloss', logging_level='Silent')
        elif model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError("Invalid model_type...")

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        pipeline.fit(X_resampled, y_resampled.iloc[:, 0])
        accuracy = pipeline.score(X_resampled, y_resampled)

        logging.info(f"Model {index + 1}/{self.nmodels} trained. Accuracy: {accuracy:.2%}")
        return {'model': pipeline, 'accuracy': accuracy}

    def fit_scores(self, balance: bool = True, nmodels: Optional[int] = None, n_jobs: int = 1,
                   model_type: str = 'linear', max_iter: int = 100):
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
                                       [(i, self.X, self.y, self.model_type, balance, max_iter) for i in
                                        range(nmodels)])
            for res in results:
                self.models.append(res['model'])
                self.model_accuracy.append(res['accuracy'])
            logging.info(f"Average Accuracy:{np.mean(self.model_accuracy):.2%} ")
        else:
            result = self.fit_model(0, self.X, self.y, self.model_type, balance, max_iter)
            self.models.append(result['model'])
            self.model_accuracy.append(result['accuracy'])
            logging.info(f"Accuracy:{round(self.model_accuracy[0] * 100, 2)}%")

    def predict_scores(self):
        model_preds = []
        for m in self.models:
            preds = m.predict_proba(self.X)[:, 1]
            model_preds.append(preds)

        model_preds = np.array(model_preds)
        scores = np.mean(model_preds, axis=0)
        self.data['scores'] = scores

    def match(self, threshold: float = 0.001, nmatches: int = 1, method: str = 'min',
              max_rand: int = 10, replacement: bool = False):
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

        test_scores = self.data[self.data[self.yvar] == True][['scores']].sort_values('scores').reset_index()
        ctrl_scores = self.data[self.data[self.yvar] == False][['scores']].sort_values('scores').reset_index()

        test_indices = test_scores['index'].values
        test_scores_values = test_scores['scores'].values.reshape(-1, 1)
        ctrl_indices = ctrl_scores['index'].values
        ctrl_scores_values = ctrl_scores['scores'].values.reshape(-1, 1)

        # Initialize NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=nmatches, radius=threshold, algorithm='ball_tree').fit(ctrl_scores_values)

        # Find neighbors within the threshold
        distances, indices = nbrs.radius_neighbors(test_scores_values)

        matched_records = []
        current_match_id = 0
        used_ctrl_indices = set() if not replacement else None

        for i, neighbors in enumerate(indices):
            if len(neighbors) == 0:
                continue  # No match found within threshold
            if method == 'min':
                # Sort neighbors by distance
                sorted_neighbors = neighbors[np.argsort(distances[i])]
                selected = []
                for neighbor in sorted_neighbors:
                    ctrl_idx = ctrl_indices[neighbor]
                    if not replacement and ctrl_idx in used_ctrl_indices:
                        continue
                    selected.append(ctrl_idx)
                    if not replacement:
                        used_ctrl_indices.add(ctrl_idx)
                    if len(selected) == nmatches:
                        break
                if selected:
                    for ctrl_idx in selected:
                        matched_records.append({
                            'test_index': test_indices[i],
                            'control_index': ctrl_idx,
                            'match_id': current_match_id
                        })
                    current_match_id += 1
            elif method == 'random':
                # Randomly select up to nmatches from the neighbors
                possible = list(neighbors)
                if not replacement:
                    possible = [n for n in possible if ctrl_indices[n] not in used_ctrl_indices]
                if len(possible) == 0:
                    continue
                select = min(nmatches, len(possible))
                selected = np.random.choice(possible, size=select, replace=False).tolist()
                for neighbor in selected:
                    ctrl_idx = ctrl_indices[neighbor]
                    matched_records.append({
                        'test_index': test_indices[i],
                        'control_index': ctrl_idx,
                        'match_id': current_match_id
                    })
                    if not replacement:
                        used_ctrl_indices.add(ctrl_idx)
                    current_match_id += 1
            else:
                raise ValueError("Invalid method parameter, use ('random', 'min')")

        # Convert matched_records to DataFrame
        if matched_records:
            matched_df = pd.DataFrame(matched_records)
            matched_test = self.data.loc[matched_df['test_index']].copy()
            matched_ctrl = self.data.loc[matched_df['control_index']].copy()

            # Assign match_id and record_id
            matched_test['match_id'] = matched_df['match_id'].values
            matched_ctrl['match_id'] = matched_df['match_id'].values
            matched_test['record_id'] = matched_test.index
            matched_ctrl['record_id'] = matched_ctrl.index

            # Combine matched data
            self.matched_data = pd.concat([matched_test, matched_ctrl], ignore_index=True)
        else:
            self.matched_data = pd.DataFrame(columns=self.data.columns.tolist() + ['match_id', 'record_id'])

    def plot_scores(self):
        """
        Plots the distribution of propensity scores before matching between
        our test and control groups
        """
        assert 'scores' in self.data.columns, \
            "Propensity scores haven't been calculated, use Matcher.predict_scores()"
        sns.kdeplot(self.data[self.data[self.yvar] == 0].scores, label='Control', fill=True)
        sns.kdeplot(self.data[self.data[self.yvar] == 1].scores, label='Test', fill=True)
        plt.legend(loc='upper right')
        plt.xlim((0, 1))
        plt.title("Propensity Scores Before Matching")
        plt.ylabel("Density")
        plt.xlabel("Scores")
        plt.show()

    def prop_test(self, col: str) -> Optional[dict]:
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
            return None

    def compare_continuous(self, save: bool = False, return_table: bool = False, plot_result: bool = True) -> Optional[pd.DataFrame]:
        """
        Plots the ECDFs for continuous features before and
        after matching. Each chart title contains test results
        and statistics to summarize how similar the two distributions
        are (we want them to be close after matching).

        Tests performed:
        Kolmogorov-Smirnov Goodness of fit Test (KS-test)
            This test statistic is calculated on 1000
            permuted samples of the data, generating
            an empirical p-value.  See pysmatch.functions.ks_boot()
            This is an adaptation of the ks.boot() method in
            the R "Matching" package
            https://www.rdocumentation.org/packages/Matching/versions/4.9-2/topics/ks.boot
        Chi-Square Distance:
            Similarly this distance metric is calculated on
            1000 permuted samples.
            See pysmatch.functions.grouped_permutation_test()

        Other included Stats:
        Standardized mean and median differences
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
                    ax1.set_title(f'''
                    ECDF for {col} before Matching
                    KS p-value: {ksb}
                    Grouped Perm p-value: {pb}
                    Std. Median Difference: {std_diff_med_before}
                    Std. Mean Difference: {std_diff_mean_before}
                    ''')
                    ax2.plot(xca.x, xca.y, label='Control', color=self.control_color)
                    ax2.plot(xta.x, xta.y, label='Test', color=self.test_color)
                    ax2.set_title(f'''
                    ECDF for {col} after Matching
                    KS p-value: {ksa}
                    Grouped Perm p-value: {pa}
                    Std. Median Difference: {std_diff_med_after}
                    Std. Mean Difference: {std_diff_mean_after}
                    ''')
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

    def compare_categorical(self, return_table: bool = False, plot_result: bool = True) -> Optional[pd.DataFrame]:
        """
        Plots the proportional differences of each enumerated
        discrete column for test and control.
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

        def prep_plot(data: pd.DataFrame, var: str, colname: str) -> pd.DataFrame:
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
                if test_results_i is not None:
                    test_results.append(test_results_i)
                if plot_result and test_results_i is not None:
                    # plotting
                    df.plot.bar(alpha=.8)
                    plt.title(title_str.format(col, test_results_i["before"],
                                               test_results_i["after"]))
                    lim = max(.09, abs(df).max().max()) + .01
                    plt.ylim((-lim, lim))
                    plt.show()
        return pd.DataFrame(test_results)[['var', 'before', 'after']] if return_table else None

    def prep_prop_test(self, data: pd.DataFrame, var: str) -> List[List[int]]:
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
        counts = data.groupby([var, self.yvar]).size().unstack(fill_value=0)
        # Ensure both classes 0 and 1 are present
        if 0 not in counts.columns:
            counts[0] = 0
        if 1 not in counts.columns:
            counts[1] = 0
        counts = counts[[0, 1]]
        ctable = counts.values.tolist()
        return ctable

    def prop_retained(self) -> float:
        """
        Returns the proportion of data retained after matching
        """
        return len(self.matched_data[self.matched_data[self.yvar] == self.minority]) * 1.0 / \
            len(self.data[self.data[self.yvar] == self.minority])

    def tune_threshold(self, method: str, nmatches: int = 1, rng: np.ndarray = np.arange(0, .001, .0001)):
        """
        Matches data over a grid to optimize threshold value and plots results.

        Parameters
        ----------
        method : str
            Method used for matching (use "random" or "min" for this method)
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
        plt.xticks(rng, rotation=90)
        plt.show()

    def record_frequency(self) -> pd.DataFrame:
        """
        Calculates the frequency of specific records in
        the matched dataset

        Returns
        -------
        pd.DataFrame()
            Frequency table of the number records
            matched once, twice, ..., etc.
        """
        freqs = self.matched_data['match_id'].value_counts().reset_index()
        freqs.columns = ['freq', 'n_records']
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