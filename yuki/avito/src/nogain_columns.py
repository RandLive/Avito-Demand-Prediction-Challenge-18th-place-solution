import json
with open("../tmp/feature_importance_lgb_nooofohe.json") as f:
    dic = json.load(f)

l_0 = []
for k, v in dic.items():
    if v<1:
        l_0.append(k)
f =open("../tmp/no_gain_features.txt", "w")
for l in l_0:
    f.write(l)
    f.write("\n")
f.close()
