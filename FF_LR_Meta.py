#FF
#Added normalize_desc()
#Added preliminary lowercase()
import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import lightgbm as lgb
import sklearn
from mlxtend.regressor import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from datetime import datetime

print('Start Runnin at ', datetime.now() )

sklearn.set_config(True) #faster option

NUM_BRANDS = 3000 #ff1312 4000 < 3000 > 2500
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 40000



def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

def handle_no_description(dataset):
    nodesc = dataset['item_description'] == 'No description yet'
    dataset.loc[nodesc, 'item_description'] = dataset.loc[nodesc, 'name']

def normalize_desc(dataset):
    dataset.item_description = dataset.item_description.str.replace(' 14 k ', ' 14k ')
    dataset.item_description = dataset.item_description.str.replace(' 14kt ', ' 14k ')
    dataset.item_description = dataset.item_description.str.replace(' power bank ', ' powerbank ')
    dataset.item_description = dataset.item_description.str.replace(' karats ', ' carats ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 7 ', ' iphone7 ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 6s ', ' iphone6s ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 6 ', ' iphone6 ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 5 ', ' iphone5 ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 5s ', ' iphone5s ')
    dataset.name = dataset.name.str.replace(' iphone 7 ', ' iphone7 ')
    dataset.name = dataset.name.str.replace(' iphone 6s ', ' iphone6s ')
    dataset.name = dataset.name.str.replace(' iphone 6 ', ' iphone6 ')
    dataset.name = dataset.name.str.replace(' iphone 5 ', ' iphone5 ')
    dataset.name = dataset.name.str.replace(' iphone 5s ', ' iphone5s ')
    dataset.item_description = dataset.item_description.str.replace(' 64g ', ' 64gb ')
    dataset.item_description = dataset.item_description.str.replace(' 32g ', ' 32gb ')
    dataset.item_description = dataset.item_description.str.replace(' iphone case ', ' iphonecase ')



start_time = time.time()

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')
print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

nrow_train = train.shape[0]
y = np.log1p(train["price"])
merge: pd.DataFrame = pd.concat([train, test])
submission: pd.DataFrame = test[['test_id']]

del train
del test
gc.collect()

handle_missing_inplace(merge)
print('[{}] Finished to handle missing'.format(time.time() - start_time))

cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))

merge.item_description = merge.item_description.str.lower()

normalize_desc(merge)
print('[{}] Finished to normalize'.format(time.time() - start_time))

#    handle_no_description(merge)
#    print('[{}] Finished to copy names to missing desc'.format(time.time() - start_time))

cv = CountVectorizer(min_df=NAME_MIN_DF, ngram_range=(1, 3)) #ff added
X_name = cv.fit_transform(merge['name'])
X_name = np.log1p(X_name)
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])
X_category = np.log1p(X_category)
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))
#FF
from nltk.corpus import stopwords
interesting_words = ['new', 'perfect', 'fit', 'used', #'super', 'cute', 'excellent',
                 'great', 'retail', '[rm]', 'never used', 'bundle', #'diamond', 'ruby',
             'platinum', 'gold', 'set', 'case', 'unused', 'unopened', 'sealed',
         'with authenticity', 'china glaze', 'authenticity card', 'complete series', #new
         'camera bag', 'no box', 'original receipt', 'day usps', 'not pictured',
         'serious offers', 'box factory' ]
X_intcol = pd.DataFrame()
for word in interesting_words:
    X_intcol[word] = merge['item_description'].apply(lambda x : word in x)

X_des = merge['item_description'].apply(lambda x: len(x)).astype('float32')
X_des = X_des[:, np.newaxis]
scaler = MaxAbsScaler()
X_des = scaler.fit_transform(X_des)

ignore_words = ['cant', 'ask', 'size', 'inch', 'inches', 'already', 'inside', 'easy']
stop = stopwords.words('english') + ignore_words
#FF
#FF version 5 n_features=2**18 --> n_features=2**17
hv = HashingVectorizer(input='content', stop_words= stop, n_features=2**17,
                        lowercase=False,
                        decode_error='ignore')
X_description = hv.transform(merge['item_description'])
print('[{}] Finished Hash vectorize `item_description`'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, 
                        X_name, X_intcol, X_des)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))

X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

#%%
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


R3 = Ridge(solver="saga", fit_intercept=False, random_state=101, 
                copy_X=True,
                max_iter=200,
                tol=0.005,
                alpha = 0.5)
scaler = MaxAbsScaler()
modelR3 = Pipeline([('scaler', scaler), ('Ridge', R3)])
#X_scale = scaler.fit_transform(X)
#X_test_scale = scaler.transform(X_test)
#modelR3.fit(X_scale, y)
#print('[{}] Finished to train ridge SAGA'.format(time.time() - start_time))
#predsR3 = modelR3.predict(X=X_test_scale)
#print('[{}] Finished to predict ridge SAGA'.format(time.time() - start_time))
modelR1 = Ridge(solver="sag", fit_intercept=True, random_state=2, alpha=4, 
                tol=0.0006,
                max_iter=800)

modelR2 = Ridge(solver="sag", fit_intercept=True, random_state=145, 
                alpha = 0.4)

params = {'application': 'regression',
 'feature_fraction': 0.8,
 'learning_rate': 0.56,
 'max_depth': 5,
 'metric': 'RMSE',
 'n_estimators': 7000,
 'n_jobs': 4,
 'num_leaves': 40,
 'random_state': 144,
 'max_bin': 8192,
 'verbosity': 10
 }

params2 = {'application': 'regression',
 'learning_rate': 0.85,
 'max_depth': 3,
 'metric': 'RMSE',
 'n_estimators': 2000,
 'n_jobs': 4,
 'num_leaves': 110,
 'random_state': 144,
 'max_bin': 8192,
 'verbosity': 10
 }

modelL1 = lgb.LGBMRegressor(**params)
modelL2 = lgb.LGBMRegressor(**params2)
metaregr = Ridge(solver="sag", max_iter=300)
stregr = StackingRegressor(regressors=[modelR1, modelR3, modelL1, modelL2], 
                           meta_regressor=metaregr, verbose=10)
stregr.fit(X, y)
print('Weights/Iter of Regressors=', stregr.coef_)
#preds = stregr.predict(X)
#print('RMSE=', mean_squared_error(y, preds)**0.5)

pred_test = stregr.predict(X_test)

submission['price'] = np.expm1(pred_test)
submission.to_csv("FF_LR_Meta.csv", index=False)

#==============================================================================
# GridSearch Cross Validation
# paramsGSCV = {
#           'ridge__alpha': [0.4, 1.0],
#           'lgbmregressor__max_depth': [4, 5],
#           'meta-ridge__alpha': [1.0]
#         }
# stregr.get_params().keys() #Print list of available parameters
# grid = GridSearchCV(estimator=stregr, 
#                     param_grid=paramsGSCV, 
#                     cv=4,
#                     refit=True,
#                     verbose=10
#        )
# grid.fit(X, y)
# 
#==============================================================================

#FF errors of the models
#    from sklearn.metrics import mean_squared_error
#    predTL1 = modelL1.predict(X)
#    predTL2 = modelL2.predict(X)
#    predTR1 = modelR1.predict(X)
#    predTR2 = modelR2.predict(X)
#    predall = predTL1*0.55 + predTL2*0.15 + predTR1*0.15 + predTR2*0.15
#    print("MSE:", 
#        "L1=", mean_squared_error(y,predTL1),
#        "L2=", mean_squared_error(y,predTL2),
#        "R1=", mean_squared_error(y,predTR1),
#        "R2=", mean_squared_error(y,predTR2),
#        "\nALL=", mean_squared_error(y,predall)
#        )
#
#    high_err = merge.loc[(y-predall) > 2.0]
#    low_err = merge.loc[(y-predall) < -2.0]
#
#    print("Much Higher priced items: ", len(high_err))
#    print("Much Lower priced items: ", len(low_err))

#Fitting 4 regressors...
#Fitting regressor1: ridge (1/4)
#Ridge(alpha=4, copy_X=True, fit_intercept=True, max_iter=800, normalize=False,
#   random_state=2, solver='sag', tol=0.0006)
#Fitting regressor2: pipeline (2/4)
#Pipeline(memory=None,
#     steps=[('scaler', MaxAbsScaler(copy=True)), ('Ridge', Ridge(alpha=0.5, copy_X=True, fit_intercept=False, max_iter=200,
#   normalize=False, random_state=101, solver='saga', tol=0.005))])
#Fitting regressor3: lgbmregressor (3/4)
#LGBMRegressor(application='regression', boosting_type='gbdt',
#       colsample_bytree=1.0, feature_fraction=0.8, learning_rate=0.56,
#       max_bin=255, max_depth=5, metric='RMSE', min_child_samples=10,
#       min_child_weight=5, min_split_gain=0.0, n_estimators=7000, n_jobs=4,
#       num_leaves=40, objective=None, random_state=144, reg_alpha=0.0,
#       reg_lambda=0.0, silent=True, subsample=1.0, subsample_for_bin=50000,
#       subsample_freq=1, verbosity=10)
#Fitting regressor4: lgbmregressor (4/4)
#LGBMRegressor(application='regression', boosting_type='gbdt',
#       colsample_bytree=1.0, learning_rate=0.85, max_bin=255, max_depth=3,
#       metric='RMSE', min_child_samples=10, min_child_weight=5,
#       min_split_gain=0.0, n_estimators=2000, n_jobs=4, num_leaves=110,
#       objective=None, random_state=144, reg_alpha=0.0, reg_lambda=0.0,
#       silent=True, subsample=1.0, subsample_for_bin=50000,
#       subsample_freq=1, verbosity=10)
#
#Weights/Iter of Regressors= [-1.01447139  1.16501091  0.90284645 -0.05363235]
#
#RMSE= 0.353282526823