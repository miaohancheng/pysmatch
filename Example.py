# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

from pysmatch.Matcher import Matcher
import pandas as pd
import numpy as np

path = "misc/loan.csv"
data = pd.read_csv(path)


test = data[data.loan_status == "Default"]
control = data[data.loan_status == "Fully Paid"]

test['loan_status'] = 1
control['loan_status'] = 0

m = Matcher(test, control, yvar="loan_status", exclude=[])

np.random.seed(20240919)

# ============ (1) Noraml train (Without optuna) =============
# m.fit_scores(balance=True, nmodels=10, n_jobs=3, model_type='knn')
# m.fit_scores(balance=True, nmodels=10, n_jobs=3, model_type='tree', max_iter=100)
m.fit_scores(balance=True, nmodels=10, n_jobs=3, model_type='linear', max_iter=200)

# ============ (2) Utilize optuna (Only train one best model) =============
# m.fit_scores(
#     balance=True,
#     model_type='tree',
#     max_iter=200,
#     use_optuna=True,
#     n_trials=15
# )

m.predict_scores()
m.plot_scores()

m.tune_threshold(method='random')
m.match(method="min", nmatches=1, threshold=1, replacement=False)
m.plot_matched_scores()
freq_df = m.record_frequency()
m.assign_weight_vector()
print("top 6 matched data")
print(m.matched_data.sort_values("match_id").head(6))

categorical_results = m.compare_categorical(return_table=True, plot_result=True)
print(categorical_results)

cc = m.compare_continuous(return_table=True, plot_result=True)
print(cc)