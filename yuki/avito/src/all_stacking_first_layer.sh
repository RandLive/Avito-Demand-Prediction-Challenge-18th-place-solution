# lgbm
# lgbm_nooof
# lgbm binary classification
# lgbm high prob classification
# lgbm regression
# xgb
# xgb_nooof
# MLP
python3 lgbm_cv.py > logs/lgbm_cv.log
python3 lgbm_cv_no_oof.py > logs/lgbm_cv_no_oof.log
python3 oof_lgbm_classification.py --feat all > logs/lgbm_cv_all.log
python3 oof_lgbm_classification.py --feat nooof > logs/lgbm_cv_nooof.log
# python3 xgb_cv.py > logs/xgb_cv.log
# python3 xgb_cv_no_oof.py > logs/xgb_cv_no_oof.log
