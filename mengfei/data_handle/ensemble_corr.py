# -*- coding: utf-8 -*-

# https://www.kaggle.com/the1owl/blend-trolling

import pandas as pd

b1 = pd.read_csv('ml_lgb_sub_1.csv').rename(columns={'deal_probability':'dp1'})
b2 = pd.read_csv('ml_lgb_sub_2.csv').rename(columns={'deal_probability':'dp2'})

b1 = pd.merge(b1, b2, how='left', on='item_id')

b1['deal_probability'] = (b1['dp1'] * 0.5) + (b1['dp2'] * 0.5)
b1[['item_id','deal_probability']].to_csv('ml_lgb_sub_0_1.csv', index=False)

print('correlation between models outputs')


blend_results = pd.concat([ b1['dp1'], b1['dp2'] ],axis=1)
print(blend_results.corr())