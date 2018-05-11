# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(ML)s
lb 0.228
"""

import pandas as pd
import numpy as np

# sub0 = pd.read_csv('subm/combine_submission2018-02-27-11-34_0.5_0.5_09862.csv')
sub0 = pd.read_csv('subm/xgb_tfidf0.225.csv')
sub1 = pd.read_csv('subm/xgb_tfidf0.226.csv')#
sub2 = pd.read_csv('subm/xgb_tfidf0.2256.csv')#
# sub0 = pd.read_csv('subm/combine_09869_0985x2018-03-03-22-01_0.55_0.44999999999999996.csv')#0.9862+0.0961
sub3 = pd.read_csv('subm/xgb_tfidf0.2258.csv')#0
# combine_09869_0985x2018-03-03-22-01_0.55_0.44999999999999996
sub4 = pd.read_csv('subm/xgb_tfidf0.2254.csv')#0
# sub4 = pd.read_csv('subm/superblend_1_09854.csv') #

sub5 = pd.read_csv('subm/lgsub02251.csv')
sub6= pd.read_csv('subm/xgb_tfidf0.2255.csv') #
sub7 = pd.read_csv('subm/lgsub02252.csv')
sub8= pd.read_csv('subm/lgsub02254.csv')
sub9= pd.read_csv('subm/lgsub02251_v2.csv')

sub10 = pd.read_csv('subm/xgb_tfidf0.2247.csv')
sub11= pd.read_csv('subm/xgb_tfidf0.2249.csv')
sub12= pd.read_csv('subm/lgsub_0225.csv')

sub13 = pd.read_csv('subm/xgb_tfidf0.2248.csv')
sub14 = pd.read_csv('subm/lgsub0.2250.csv')
sub15 = pd.read_csv('subm/lgsub02250.csv')

sub16 = pd.read_csv('subm/xgb_tfidf0.2245.csv')
sub17 = pd.read_csv('subm/xgb_tfidf0.2245_v2.csv')
sub18 = pd.read_csv('subm/lgsub02244.csv')
sub19 = pd.read_csv('subm/xgb_tfidf0.2245_v3.csv')
label_cols = ['deal_probability']

p_res = sub0

for col in label_cols:
    # p_res[col] = 0.5*sub0[col]+0.16*sub1[col]+0.14*sub3[col]+0.2*sub4[col]
    # p_res[col] = 0.23 * sub0[col] + 0.23 * sub1[col]+ 0.18 * sub3[col] + 0.18 * sub3[col] + 0.18 * sub4[col]
    # p_res[col] = 0.23 * sub0[col] + 0.23 * sub1[col] + 0.14 * sub3[col] + 0.14 * sub3[col] + 0.13 * sub4[col]+0.13 * sub5[col]
    # p_res[col] =  (sub0[col] +  sub1[col] +  sub2[col]+sub3[col]+sub4[col])/5
    #0.2236
    # p_res[col] = (sub0[col] + sub1[col] + sub2[col] + sub3[col] + sub4[col]+ sub5[col] + sub6[col]) / 7
    # p_res[col] = (sub0[col] + sub1[col] + sub2[col] + sub3[col] + sub4[col] + sub5[col] + sub6[col]+ sub7[col] + sub8[col]) / 9
    # p_res[col] = (sub0[col] + sub1[col] + sub2[col] + sub3[col] + sub4[col] + sub5[col] + sub6[col] + sub7[col] + sub8[col]+ sub9[col]) / 10
    # p_res[col] = (sub0[col] + sub1[col] + sub2[col] + sub3[col] + sub4[col] + sub5[col] + sub6[col] + sub7[col] + sub8[
    #     col] + sub9[col]+ sub10[col]+ sub11[col]+ sub12[col]) / 13
    # p_res[col] = (sub0[col] + sub1[col] + sub2[col] + sub3[col] + sub4[col] + sub5[col] + sub6[col] + sub7[col] + sub8[
    #     col] + sub9[col] + sub10[col] + sub11[col] + sub12[col]+sub13[col]+ sub14[col]+sub15[col]) / 16
    # p_res[col] = (sub0[col] + sub1[col] + sub2[col] + sub3[col] + sub4[col] + sub5[col] + sub6[col] + sub7[col] + sub8[
    #     col] + sub9[col] + sub10[col] + sub11[col] + sub12[col] + sub13[col] + sub14[col] + sub15[col]+ sub16[col] + sub17[col]) / 18
    p_res[col] = (sub0[col] + sub1[col] + sub2[col] + sub3[col] + sub4[col] + sub5[col] + sub6[col] + sub7[col] + sub8[
        col] + sub9[col] + sub10[col] + sub11[col] + sub12[col] + sub13[col] + sub14[col] + sub15[col] + sub16[col] +
                  sub17[col]+ sub18[col] + sub19[col]) / 20

p_res.to_csv('subm/xgb_tfidf_20fold.csv', index=False)