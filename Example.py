import warnings
warnings.filterwarnings('ignore')
from pysmatch.Matcher import Matcher
import pandas as pd
import numpy as np

print('get data')
path = "misc/loan.csv"
data = pd.read_csv(path)

test = data[data.loan_status == "Default"]
control = data[data.loan_status == "Fully Paid"]
print('get data')

test['loan_status'] = 1
control['loan_status'] = 0


m = Matcher(test, control, yvar="loan_status", exclude=[])


# for reproducibility
np.random.seed(20170925)

m.fit_scores(balance=True, nmodels=5,n_jobs=1,model_type='tree')


m.predict_scores()

m.plot_scores()
m.tune_threshold(method='random')
m.match(method="min", nmatches=1, threshold=0.0005)

m.record_frequency()
m.assign_weight_vector()
m.matched_data.sort_values("match_id").head(6)
categorical_results = m.compare_categorical(return_table=True)
print(categorical_results)
cc = m.compare_continuous(return_table=True)
print(cc)