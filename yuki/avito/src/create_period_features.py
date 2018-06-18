
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gc
from utils import *

# In[3]:


train_active = pd.read_csv("../input/train_active.csv")
test_active = pd.read_csv("../input/test_active.csv")
train_periods = pd.read_csv("../input/periods_train.csv", parse_dates=['date_from', 'date_to'])
test_periods = pd.read_csv("../input/periods_test.csv", parse_dates=['date_from', 'date_to'])


# In[7]:


all_periods = pd.concat([
    train_periods,
    test_periods
])

del train_periods
del test_periods
gc.collect()

# all_periods['days_up'] = (all_periods['date_to'] - all_periods['date_from']).dt.days


# In[12]:


all_periods['days_up']  = all_periods['date_to'].subtract(all_periods['date_from']).dt.days


# In[13]:


all_periods.head()


# In[14]:


all_periods = all_periods.groupby("item_id").days_up.mean().reset_index()


# In[15]:


all_periods.head()


# In[ ]:


active = pd.concat([train_active, test_active])
del train_active, test_active;gc.collect()
active.drop_duplicates(['item_id'], inplace=True)


# In[ ]:


df = pd.merge(active, all_periods, on="item_id", how="left")


# In[ ]:


del active, all_periods; gc.collect()


# In[ ]:


target_columns = ["category_name", "item_seq_number"]


# In[ ]:



train = pd.read_csv("../input/train.csv", usecols=["category_name", "item_seq_number"])
test = pd.read_csv("../input/test.csv", usecols=["category_name", "item_seq_number"])
n_train = train.shape[0]
all_df = pd.concat([train, test])
del train, test; gc.collect()
# df_cate = pd.DataFrame()
# categories = ["category_name_daysup_mean", "category_name_daysup_std","category_name_daysup_max", "category_name_daysup_min"]
# df_cate[categories[0]] = df.groupby("category_name").days_up.mean()
# df_cate[categories[1]] = df.groupby("category_name").days_up.std()
# df_cate[categories[2]] = df.groupby("category_name").days_up.max()
# df_cate[categories[3]] = df.groupby("category_name").days_up.min()
# df_cate = df_cate.reset_index()
# df_out = pd.merge(all_df, df_cate, on="category_name", how="left")
# to_parquet(df_out.iloc[:n_train,:][categories], "../features/fe_daysup_category_name_train.parquet")
# to_parquet(df_out.iloc[n_train:,:][categories], "../features/fe_daysup_category_name_test.parquet")

categories = ["item_seq_number_daysup_mean", "item_seq_number_daysup_std","item_seq_number_daysup_max", "item_seq_number_daysup_min"]
df_cate = pd.DataFrame()
df_cate[categories[0]] = df.groupby("item_seq_number").days_up.mean()
df_cate[categories[1]] = df.groupby("item_seq_number").days_up.std()
df_cate[categories[2]] = df.groupby("item_seq_number").days_up.max()
df_cate[categories[3]] = df.groupby("item_seq_number").days_up.min()
df_cate = df_cate.reset_index()
df_out = pd.merge(all_df, df_cate, on="item_seq_number", how="left")
to_parquet(df_out.iloc[:n_train,:][categories], "../features/fe_daysup_item_seq_number_train.parquet")
to_parquet(df_out.iloc[n_train:,:][categories], "../features/fe_daysup_item_seq_number_test.parquet")
