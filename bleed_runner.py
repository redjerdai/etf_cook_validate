#

#
import numpy
import pandas
from matplotlib import pyplot

#

#
"""
Data
"""
_data = pandas.read_excel("./data/series.xlsx")
_data = _data.set_index("date")

data = _data.copy().sort_index()

data.index = data.index + pandas.offsets.MonthBegin()

d = './data/stats.xlsx'
stats = pandas.read_excel(d)
stats = stats.set_index('date')

data = pandas.concat((data, stats), axis=1)
min_ix_obs = data.index.min()
min_ix_nna = data.dropna().index.min()
print(min_ix_obs)
data = data.query("index >= '{0}'".format(min_ix_nna))

data_pct = data.pct_change().dropna()

data_pct_lagged = data_pct.copy()
data_pct_lagged[[x + '_LAG1' for x in data_pct.columns.values]] = data_pct.shift(periods=1)
data_pct_lagged[[x + '_LAG2' for x in data_pct.columns.values]] = data_pct.shift(periods=2)
data_pct_lagged[[x + '_LAG3' for x in data_pct.columns.values]] = data_pct.shift(periods=3)
data_pct_lagged[[x + '_LAG4' for x in data_pct.columns.values]] = data_pct.shift(periods=4)
data_pct_lagged = data_pct_lagged.dropna()

"""
X = data_pct.values[1:, [0, 1]]
Y = data_pct.values[:-1, [0, 1]]
"""
excluded = ['FXT', 'EFA', 'EEM']
markers = [x for x in data.columns.values if x not in excluded]
cols_x = [x for x in data_pct_lagged if any([y in x for y in markers])]
X = data_pct_lagged.loc[:, cols_x].values[1:, :]
Y = data_pct_lagged.values[:-1, [0, 1]]
X_ = data_pct_lagged.values[1:, [0, 1]]
Y_ = data_pct_lagged.values[:-1, [0, 1]]

tt = data_pct_lagged.index.values[:-1]

thresh = 100

# Varloss "sola": 2L Tree
"""
predicted = numpy.full(shape=(Y.shape[0],), fill_value=numpy.nan, dtype=Y.dtype)

predicted[(X[:, cols_x.index("10m2YY_LAG1")] <= -2.5) * (X[:, cols_x.index("TLT_LAG4")] <= -0.028)] = 0.945
predicted[(X[:, cols_x.index("10m2YY_LAG1")] <= -2.5) * ~(X[:, cols_x.index("TLT_LAG4")] <= -0.028)] = 0.99
predicted[~(X[:, cols_x.index("10m2YY_LAG1")] <= -2.5) * (X[:, cols_x.index("10YGB_LAG4")] <= 0.033)] = 0.468
predicted[~(X[:, cols_x.index("10m2YY_LAG1")] <= -2.5) * ~(X[:, cols_x.index("10YGB_LAG4")] <= 0.033)] = 0.594

predicted = predicted.reshape(-1, 1)
predicted = numpy.concatenate((predicted, 1 - predicted), axis=1)

result_dynamics_vst = (predicted * Y).sum(axis=1)
result_cum_vst = (1 + result_dynamics_vst).cumprod()
"""
# Varloss "paty": 2L Tree

predicted = numpy.full(shape=(Y[thresh:].shape[0],), fill_value=numpy.nan, dtype=Y.dtype)

predicted[(X[thresh:, cols_x.index("TLT_LAG1")] <= 0.017) * (X[thresh:, cols_x.index("IVV_LAG1")] <= 0.004)] = 0.588
predicted[(X[thresh:, cols_x.index("TLT_LAG1")] <= 0.017) * ~(X[thresh:, cols_x.index("IVV_LAG1")] <= 0.004)] = 0.923
predicted[~(X[thresh:, cols_x.index("TLT_LAG1")] <= 0.017) * (X[thresh:, cols_x.index("IVV_LAG1")] <= 0.024)] = 0.057
predicted[~(X[thresh:, cols_x.index("TLT_LAG1")] <= 0.017) * ~(X[thresh:, cols_x.index("IVV_LAG1")] <= 0.024)] = 0.752

predicted = predicted.reshape(-1, 1)
predicted = numpy.concatenate((predicted, 1 - predicted), axis=1)

commi = 0.01
result_dynamics_vpt = (predicted * Y[thresh:]).sum(axis=1) * (1 - commi)

# this is target series; it's last value [-1] = resulting performance of the model
result_cum_vpt = (1 + result_dynamics_vpt).cumprod()
