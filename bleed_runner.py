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
cols_x = [x for x in data_pct_lagged.columns.values if 'LAG' in x and ~any([y in x for y in excluded])]
X = data_pct_lagged.loc[:, cols_x].values[1:, :]
Y = data_pct_lagged.values[:-1, [0, 1]]
X_ = data_pct_lagged.values[1:, [0, 1]]
Y_ = data_pct_lagged.values[:-1, [0, 1]]

tt = data_pct_lagged.index.values[:-1]

thresh = 100

# Varloss "paty": 2L Tree

predicted = numpy.full(shape=(Y[thresh:].shape[0],), fill_value=numpy.nan, dtype=Y.dtype)

predicted[(X[thresh:, cols_x.index("10YGB_LAG1")] <= -0.013) * (X[thresh:, cols_x.index("10YGB_LAG1")] <= -0.06)] = 0.074
predicted[(X[thresh:, cols_x.index("10YGB_LAG1")] <= -0.013) * ~(X[thresh:, cols_x.index("10YGB_LAG1")] <= -0.06)] = 0.482
predicted[~(X[thresh:, cols_x.index("10YGB_LAG1")] <= -0.013) * (X[thresh:, cols_x.index("UNEMPLJ_LAG2")] <= 0.043)] = 0.879
predicted[~(X[thresh:, cols_x.index("10YGB_LAG1")] <= -0.013) * ~(X[thresh:, cols_x.index("UNEMPLJ_LAG2")] <= 0.043)] = 0.251

predicted = predicted.reshape(-1, 1)
predicted = numpy.concatenate((predicted, 1 - predicted), axis=1)

commi = 0.01
result_dynamics_vpt = (predicted * Y[thresh:]).sum(axis=1) * (1 - commi)
result_cum_vpt = (1 + result_dynamics_vpt).cumprod()
