import pandas as pd
import numpy as np

# df = pd.read_csv("../output/blendall.csv")
# probs = set(pd.read_csv("../input/train.csv", usecols=["deal_probability"])["deal_probability"].values)
# def find_nearest(p):
#     k = 10000
#     sol = 0
#     for pro in probs:
#         tmp = np.abs(p-pro)
#         if tmp < k:
#             k = tmp
#             sol = pro
#     return sol
#
# df["deal_probability"] = df["deal_probability"].apply(lambda x: find_nearest(x))
# df.to_csv("../output/blendall_postprocessed.csv", index=False)

df = pd.read_csv("../output/blendall.csv")
probs = pd.read_csv("../input/train.csv", usecols=["deal_probability"])["deal_probability"].value_counts().to_dict()

def find_nearest(p):
    k = 10000
    sol = 0
    for pro in probs:
        tmp = np.abs(p-pro)
        if tmp < k:
            k = tmp
            sol = pro
    return sol

df["deal_probability"] = df["deal_probability"].apply(lambda x: find_nearest(x))
df.to_csv("../output/blendall_postprocessed.csv", index=False)
