import pandas as pd

b1 = pd.read_csv('ml_lgb_sub_1.csv').rename(columns={'deal_probability':'dp1'})
b2 = pd.read_csv('ml_lgb_sub_2.csv').rename(columns={'deal_probability':'dp2'})
b3 = pd.read_csv('ml_lgb_sub_3.csv').rename(columns={'deal_probability':'dp3'})
b4 = pd.read_csv('ml_lgb_sub_4.csv').rename(columns={'deal_probability':'dp4'})
b5 = pd.read_csv('ml_lgb_sub_5.csv').rename(columns={'deal_probability':'dp5'})

b1 = pd.merge(b1, b2, how='left', on='item_id')
b1 = pd.merge(b1, b3, how='left', on='item_id')
b1 = pd.merge(b1, b4, how='left', on='item_id')
b1 = pd.merge(b1, b5, how='left', on='item_id')


b1['deal_probability'] = (b1['dp1']+b1['dp2']+b1['dp3']+b1['dp4']+b1['dp5'])/5


b1[['item_id','deal_probability']].to_csv('lgb_test.csv', index=False)
print('correlation between models outputs')
blend_results = pd.concat([ b1['dp1'], b1['dp2'], b1['dp3'], b1['dp4'], b1['dp5']],axis=1)
print(blend_results.corr())


'''
dp1  1.000000  0.993578  0.998357
dp2  0.993578  1.000000  0.998429
dp4  0.998357  0.998429  1.000000
'''