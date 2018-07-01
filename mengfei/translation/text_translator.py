#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:19:41 2018
@author: qifeng, mengfei
"""

from textblob import TextBlob
from textblob.translate import NotTranslated
import pandas as pd
import time
from sklearn.utils import shuffle
from joblib import Parallel, delayed
import argparse
import os
import pandas as pd

NAN_WORD = "_NAN_"



print("Starting job at time:",time.time())
debug = True
print("loading data ...")
used_cols = ["item_id", "user_id"]
if debug == False:
    train_df = pd.read_csv("../input/train.csv",  parse_dates = ["activation_date"])
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv",  parse_dates = ["activation_date"])

else:
    train_df = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:100]
    test_df = pd.read_csv("../input/test.csv",  nrows=1000, parse_dates = ["activation_date"])

print("loading data done!")


def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
#        text = text.translate(to=language)
        text = text.translate(to="en")
    except NotTranslated:
        pass

    return str(text)



parser = argparse.ArgumentParser("Script for extending train dataset")
#    parser.add_argument("input/train.csv")
parser.add_argument("--languages", nargs="+", default=["en"])
parser.add_argument("--thread count", type=int, default=300)
parser.add_argument("--result-path", default="extended_data")

args = parser.parse_args()

#    train_data = pd.read_csv(args.train_file_path)
comments_list = train_df["title"].fillna(NAN_WORD).values

if not os.path.exists(args.result_path):
      os.mkdir(args.result_path)

parallel = Parallel(11, backend="threading", verbose=5)
for language in args.languages:
      print('Translate comments using "{0}" language'.format(language))
      translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)
      train_df["title_ru"] = translated_data
      
      result_path = os.path.join(args.result_path, "train_" + language + ".csv")
      train_df.to_csv(result_path, index=False)


    
tmp = translate('это красиво,  beautiful ', 'en')


def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
#        text = text.translate(to=language)
        text = text.translate(to="ru")
    except NotTranslated:
        pass

    return str(text)


print(translate(tmp, 'ru'))