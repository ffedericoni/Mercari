"""
LGBM Regression on TfIDF of text features and One-Hot-Encoded Categoricals
Featues based on Alexandu Papiu's (https://www.kaggle.com/apapiu) script: https://www.kaggle.com/apapiu/ridge-script
LGBM based on InfiniteWing's (https://www.kaggle.com/infinitewing) script: https://www.kaggle.com/infinitewing/lightgbm-example
"""
#TODO don't use dummies, but categorical features of LightGBM

import pandas as pd
import numpy as np
import scipy

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
import time
from nltk.corpus import stopwords
import gc

start = fixstart = time.time()

NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

print("Reading in Data")

df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
y_train = np.log1p(df_train["price"])

del df_train
gc.collect()

print(df.memory_usage(deep = True))

df["category_name"] = df["category_name"].fillna("Other").astype("category")
df["brand_name"] = df["brand_name"].fillna("unknown")

pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

df["item_description"] = df["item_description"].fillna("None")
df["item_condition_id"] = df["item_condition_id"].astype("category")
df["brand_name"] = df["brand_name"].astype("category")

# description contains interesting words
interesting_words = ['new', 'perfect', 'fit', 'used', 'super', 'cute', 'excellent',
                     'great', 'retail', '[rm]', 'never' ]
for word in interesting_words:
    df[word] = df['item_description'].apply(lambda x : word in x.lower())
    df[word] = df[word].astype('category')

print(df.memory_usage(deep = True))

print("Encodings")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(df["name"])

print("Category Encoders")
unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
count_category = CountVectorizer()
X_category = count_category.fit_transform(df["category_name"])


stopw = stopwords.words('english') + interesting_words + ['cant', 'ask', 'size']
print("Descp encoders")
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = stopw)
X_descp = count_descp.fit_transform(df["item_description"])

print("Brand encoders")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(df["brand_name"])

print("Dummy Encoders")
rem_cols = interesting_words + ["item_condition_id", "shipping" ]
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[rem_cols], sparse = True).values)

X = scipy.sparse.hstack((X_dummies, 
                         X_descp,
                         X_brand,
                         X_category,
                         X_name)).tocsr()

print([X_dummies.shape, X_category.shape, 
       X_name.shape, X_descp.shape, X_brand.shape])

X_train = X[:nrow_train]
X_test = X[nrow_train:]

end = time.time()
print("Time taken reading and encoding  {}.".format((end-start)))

#LGBM Training
start = time.time()

#train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, 
#                                                      test_size = 0.1, 
#                                                      random_state = 144) 
#evalset = [(train_X, train_y),(valid_X, valid_y)]
#evalset = [(valid_X, valid_y)]
lgbmodel = lgb.LGBMRegressor(
    learning_rate= 0.8,
    objective='regression',
    min_data_in_leaf=20, #minimal number of data in one leaf.
    feature_fraction=1.0, #LightGBM will randomly select part of features on each iteration if feature_fraction smaller than 1.0.
    min_split_gain=0.0, #the minimal gain to perform split
    cat_l2=10.,         #L2 regularization in categorical split
    min_data_in_bin=3,  #min number of data inside one bin, use this to avoid one-data-one-bin 
    bagging_fraction = 1,
    silent=True,
    metric='rmse',
#    train_metric=False,
#    metric_freq=10,
    n_estimators=1000,
    cat_smooth=10, #this can reduce the effect of noises in categorical features, especially for categories with few data
    max_bin=8192, #TODO try to reduce
    num_threads=3,
    two_round_loading=True #set this to true if data file is too big to fit in memory
        )

param_grid = {
#    'learning_rate': 0.8,
    'max_depth': [3, 4, 5]
#    'num_leaves': 100,
#    'max_bin':8192
}

gbm = GridSearchCV(lgbmodel, param_grid, verbose=10, scoring='rmse')

gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)

#lgbmodel.fit(X=train_X, y=train_y, eval_set=evalset, eval_names=['train', 'valid'],
#            eval_metric='rmse',
#            early_stopping_rounds=50 )

#preds = gbm.predict(X_test)
end = time.time()
print("Time taken training LGBM  {}.".format((end-start)))


#df_test["price"] = np.expm1(preds)
#df_test[["test_id", "price"]].to_csv("ff_LGBM_Ridge_3.csv", index = False)
end = time.time()
print("Time taken by the Kernel  {}.".format((end-fixstart)))
