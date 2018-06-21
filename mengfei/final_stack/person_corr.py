import pandas as pd

b1 = pd.read_csv('ml_blend_v7.csv').rename(columns={'deal_probability':'dp1'})
b2 = pd.read_csv('stacked_xgb_sub.csv').rename(columns={'deal_probability':'dp2'})

blend_results = pd.concat([ b1['dp1'], b2['dp2']],axis=1)
print(blend_results.corr())


'''
dp1  1.000000  0.993578  0.998357
dp2  0.993578  1.000000  0.998429
dp4  0.998357  0.998429  1.000000
'''