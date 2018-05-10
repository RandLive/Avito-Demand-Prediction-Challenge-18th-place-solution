{
  "cells": [
    {
      "metadata": {
        "_cell_guid": "82aa5500-8b6f-43b0-a891-d22a5d8411f8",
        "_uuid": "90bc2d99f499f06ff75da34882d638f49594958c",
        "collapsed": true,
        "trusted": false
      },
      "cell_type": "code",
      "source": "'''\n训练集 3月15日到4月7日, 测试集4月12到4月20\n\n'''\n\n# TODO1: 使用period信息。\n# TODO2: 计算各个类别物品的均值，最大值，平均值, 中值\n\nimport pandas as pd\nfrom sklearn.preprocessing import LabelEncoder\nimport gc",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_cell_guid": "41bb9c17-8c9e-452b-befc-673a6e092652",
        "_uuid": "7bc141d70316e34dc1969ddd71c9b7f6fd2ce1c8",
        "collapsed": true,
        "trusted": false
      },
      "cell_type": "code",
      "source": "debug = True\n\nprint(\"loading data ...\")\n\ndef feature_Eng_Datetime(df):\n    print('feature engineering -> datetime ...')\n    df['wday'] = df['activation_date'].dt.weekday\n    df['week'] = df['activation_date'].dt.week\n    df['dom'] = df['activation_date'].dt.day\n    df.drop('activation_date', axis=1, inplace=True)\n    return df",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_cell_guid": "27f9ae14-ade7-4be5-916c-2ab924c6253c",
        "_uuid": "14ba1b16365db2116445afd1d04d846487e9286a",
        "collapsed": true,
        "trusted": false
      },
      "cell_type": "code",
      "source": "lbl = LabelEncoder()\n\ndef feature_Eng_label_Enc(df):\n    print('feature engineering -> lable encoding ...')\n    cat_col = [\"user_id\", \"region\", \"city\", \"parent_category_name\",\n               \"category_name\", \"user_type\", \"image_top_1\",\n               # TODO: 这里还需要西考虑一下\n               \"param_1\", \"param_2\", \"param_3\"]\n    for col in cat_col:\n        df[col] = lbl.fit_transform(df[col].astype(str))\n    del cat_col;gc.collect()\n    return df\n\n\ndef feature_Eng_NA(df):\n    print('feature engineering -> handle NA ...')\n    df['price'].fillna(-1, inplace=True)\n    df.fillna('отсутствует описание', inplace=True) # google translation of 'missing discription' into Russian\n    return df\n\n\n#def feature_Eng_ON_price(df):\n#    print('feature engineering -> statistics on price ...')\n#    df['price'].fillna(-1, inplace=True)\n#    df.fillna('отсутствует описание', inplace=True) # google translation of 'missing discription' into Russian\n#    return df\n\ndef feature_time_pr(df):\n    print('Feature engineering time data!')\n    df['shelf_period'] = df['date_to'] - df['date_from']\n    df['waiting_period'] = df['date_from'] - df['activation_date']\n    df['total_period'] = df['date_to'] - df['activation_date'] \n    df.drop(['activation_date', 'date_from', 'date_from'], axis=1, inplace=True)\n\ndef drop_image_data(df):\n    print('feature engineering -> drop image data ...')\n    df.drop('image', axis=1, inplace=True)\n    return df\n    ",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_cell_guid": "884aca18-2069-4df9-b5bc-8eb948a561d9",
        "_uuid": "e044ffd8b26f867f42cc2148cb057aa392b87661",
        "collapsed": true,
        "trusted": false
      },
      "cell_type": "code",
      "source": "# load data\nif debug == False: # Run\n    train_df = pd.read_csv('../input/train.csv', index_col = \"item_id\", parse_dates = [\"activation_date\"])\n    y = train_df['deal_probability']\n    del train_df['deal_probability']; gc.collect()\n    test_df = pd.read_csv('../input/test.csv', index_col = \"item_id\", parse_dates = [\"activation_date\"])\nelse: # debug\n    train_df = pd.read_csv('../input/train.csv', index_col = \"item_id\", nrows=10000, parse_dates = [\"activation_date\"])\n    y = train_df['deal_probability']\n    del train_df['deal_probability']; gc.collect()\n    test_df = pd.read_csv('../input/test.csv', index_col = \"item_id\", nrows=10000, parse_dates = [\"activation_date\"])\n\n\ntrain_index = len(train_df)\ntest_index = len(test_df)\n\n\n# concat dataset\nfull_df = pd.concat([train_df, test_df], axis=0)\ndel train_df, test_df\ngc.collect()",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "collapsed": true,
        "_uuid": "3aa99b06516187994b5c9330c9b2f0d7178ead0d"
      },
      "cell_type": "code",
      "source": "# Load time data\nprint('Loading time data!')\n\ntrain_pr = pd.read.csv('../input/periods_train.csv', index_col = 'item_id', parse_data = ['activation_date', 'date_from', 'date_to'])\ntest_pr = pd.read.csv('../input/periods_test.csv', index_col = 'item_id', parse_data = ['activation_date', 'date_from', 'date_to'])",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "collapsed": true,
        "_uuid": "ce178ff3197ba2f13e3f0989a5c29f620f63c5fa"
      },
      "cell_type": "code",
      "source": "# Feature engineering time data\nfeature_time_pr(train_pr)\nfeature_time_pr(test_pr)\n",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
        "collapsed": true,
        "trusted": false
      },
      "cell_type": "code",
      "source": "jfeature_Eng_Datetime(full_df)\nfeature_Eng_label_Enc(full_df)\nfeature_Eng_NA(full_df)\ndrop_image_data(full_df)\n\n\nprint(full_df.info())",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",
        "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",
        "collapsed": true,
        "trusted": false
      },
      "cell_type": "code",
      "source": "",
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "language_info": {
      "name": "python",
      "version": "3.6.4",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 1
}