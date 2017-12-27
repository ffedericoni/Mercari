#FF
#Added normalize_desc()
#Added preliminary lowercase()
import gc
import time
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb
import sklearn

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
    dataset.item_description = dataset.item_description.str.replace(' iphone 6s ', ' iphone6s ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 6 ', ' iphone6 ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 5 ', ' iphone5 ')
    dataset.item_description = dataset.item_description.str.replace(' iphone 5s ', ' iphone5s ')
    dataset.item_description = dataset.item_description.str.replace(' 64g ', ' 64gb ')
    dataset.item_description = dataset.item_description.str.replace(' 32g ', ' 32gb ')

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))


def rmse_min_func(weights):
    final_prediction = 0
    for weight, prediction in zip(weights, blend_train):
        final_prediction += weight * prediction
    return np.sqrt(mean_squared_error(y, final_prediction))


start_time = time.time()

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')
print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

tr_ids = train['train_id'].values.astype(np.int32)
te_ids = test['test_id'].values.astype(np.int32)

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

cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))
#FF
from nltk.corpus import stopwords
interesting_words = ['new', 'perfect', 'fit', 'used', #'super', 'cute', 'excellent',
                 'great', 'retail', '[rm]', 'never used', 'bundle', #'diamond', 'ruby',
             'platinum', 'gold', 'set', 'case', 'unused', 'unopened', 'sealed' ]
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
                        lowercase=False)
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
folds = 5

sgd_cv_sum = 0
ridge_cv_sum = 0
lgb_cv_sum = 0
lgb2_cv_sum = 0
lgb_pred = []
lgb2_pred = []
sgd_pred = []
ridge_pred = []
lgb_fpred = []
lgb2_fpred = []
sgd_fpred = []
ridge_fpred = []

avreal = y
lgb_avpred = np.zeros(X.shape[0])
lgb2_avpred = np.zeros(X.shape[0])
sgd_avpred = np.zeros(X.shape[0])
ridge_avpred = np.zeros(X.shape[0])
idpred = tr_ids

blend_train = []
blend_test = []

#FF name change 
X_train = X
y_train = y
#FF

train_time = timer(None)
kf = KFold(n_splits=folds, random_state=1001)
for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    start_time = timer(None)
    Xtrain, Xval = X_train[train_index], X_train[val_index]
    ytrain, yval = y_train[train_index], y_train[val_index]

#RIDGE SAG 1
    model = Ridge(solver="sag", 
                fit_intercept=True, 
                random_state=2, 
                alpha=4, 
                tol=0.0006,
                max_iter=800)
    model.fit(Xtrain, ytrain)
    sgd_scores_val = model.predict(Xval)
    sgd_RMSLE = np.sqrt(mean_squared_error(yval, sgd_scores_val))
    print('\n Fold %02d Ridge SAG1 RMSLE: %.6f' % ((i + 1), sgd_RMSLE))
    sgd_y_pred = model.predict(X_test)

#RIDGE SAG 2
    model = Ridge(solver="sag", 
                  fit_intercept=True, 
                  random_state=145, 
                  alpha = 0.4)
    model.fit(Xtrain, ytrain)
    ridge_scores_val = model.predict(Xval)
    ridge_RMSLE = np.sqrt(mean_squared_error(yval, ridge_scores_val))
    print(' Fold %02d Ridge SAG2 RMSLE: %.6f' % ((i + 1), ridge_RMSLE))
    ridge_y_pred = model.predict(X_test)
#LGB1
    params = {
        'learning_rate': 0.56,
        'application': 'regression',
        'max_depth': 5,
        'num_leaves': 40,
        'verbosity': -1,
        'metric': 'RMSE',
        'feature_fraction':0.8,
        'nthread': 4
    }

    dtrain = lgb.Dataset(Xtrain, label=ytrain, max_bin=8192)
    dval = lgb.Dataset(Xval, label=yval, max_bin=8192)
    watchlist = [dtrain, dval]
    watchlist_names = ['train', 'val']

    model = lgb.train(params,
                      train_set=dtrain,
                      num_boost_round=8000,
                      valid_sets=watchlist,
                      valid_names=watchlist_names,
                      early_stopping_rounds=None,
                      verbose_eval=1000)
    lgb_scores_val = model.predict(Xval)
    lgb_RMSLE = np.sqrt(mean_squared_error(yval, lgb_scores_val))
    print(' Fold %02d LightGBM1 RMSLE: %.6f' % ((i + 1), lgb_RMSLE))
    lgb_y_pred = model.predict(X_test)
#LGB2
    params = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 110,
        'verbosity': -1,
        'metric': 'RMSE',
        'nthread': 4
    }

    model = lgb.train(params,
                      train_set=dtrain,
                      num_boost_round=3200,
                      valid_sets=watchlist,
                      valid_names=watchlist_names,
                      early_stopping_rounds=None,
                      verbose_eval=500)
    lgb2_scores_val = model.predict(Xval)
    lgb2_RMSLE = np.sqrt(mean_squared_error(yval, lgb2_scores_val))
    print(' Fold %02d LightGBM2 RMSLE: %.6f' % ((i + 1), lgb2_RMSLE))
    lgb2_y_pred = model.predict(X_test)

    del Xtrain, Xval
    gc.collect()

    timer(start_time)

    sgd_avpred[val_index] = sgd_scores_val
    ridge_avpred[val_index] = ridge_scores_val
    lgb_avpred[val_index] = lgb_scores_val
    lgb2_avpred[val_index] = lgb2_scores_val

    if i > 0:
        sgd_fpred = sgd_pred + sgd_y_pred
        ridge_fpred = ridge_pred + ridge_y_pred
        lgb_fpred = lgb_pred + lgb_y_pred
        lgb2_fpred = lgb2_pred + lgb2_y_pred
    else:
        sgd_fpred = sgd_y_pred
        ridge_fpred = ridge_y_pred
        lgb_fpred = lgb_y_pred
        lgb2_fpred = lgb2_y_pred
    sgd_pred = sgd_fpred
    ridge_pred = ridge_fpred
    lgb_pred = lgb_fpred
    lgb2_pred = lgb2_fpred
    sgd_cv_sum = sgd_cv_sum + sgd_RMSLE
    ridge_cv_sum = ridge_cv_sum + ridge_RMSLE
    lgb_cv_sum = lgb_cv_sum + lgb_RMSLE
    lgb2_cv_sum = lgb2_cv_sum + lgb2_RMSLE

timer(train_time)

sgd_cv_score = (sgd_cv_sum / folds)
ridge_cv_score = (ridge_cv_sum / folds)
lgb_cv_score = (lgb_cv_sum / folds)
lgb2_cv_score = (lgb2_cv_sum / folds)
sgd_oof_RMSLE = np.sqrt(mean_squared_error(avreal, sgd_avpred))
ridge_oof_RMSLE = np.sqrt(mean_squared_error(avreal, ridge_avpred))
lgb_oof_RMSLE = np.sqrt(mean_squared_error(avreal, lgb_avpred))
lgb2_oof_RMSLE = np.sqrt(mean_squared_error(avreal, lgb2_avpred))

print('\n Average SGD RMSLE:\t%.6f' % sgd_cv_score)
print(' Out-of-fold SGD RMSLE:\t%.6f' % sgd_oof_RMSLE)
print('\n Average Ridge RMSLE:\t%.6f' % ridge_cv_score)
print(' Out-of-fold Ridge RMSLE:\t%.6f' % ridge_oof_RMSLE)
print('\n Average LightGBM RMSLE:\t%.6f' % lgb_cv_score)
print(' Out-of-fold LightGBM RMSLE:\t%.6f' % lgb_oof_RMSLE)
print('\n Average LightGBM2 RMSLE:\t%.6f' % lgb2_cv_score)
print(' Out-of-fold LightGBM2 RMSLE:\t%.6f' % lgb2_oof_RMSLE)
sgd_score = round(sgd_oof_RMSLE, 6)
ridge_score = round(ridge_oof_RMSLE, 6)
lgb_score = round(lgb_oof_RMSLE, 6)
lgb2_score = round(lgb2_oof_RMSLE, 6)

sgd_mpred = sgd_pred / folds
ridge_mpred = ridge_pred / folds
lgb_mpred = lgb_pred / folds
lgb2_mpred = lgb2_pred / folds

#%%
# This procedure finds linear blending weights from out-of-fold predictions by simple minimization. It doesn't take very long to complete, and I would normally use N-fold validation with the same folds as above. However, for the purpose of this competition the gain from doing so is negligible compared to extra time it takes.
# 
# It may be worth your while to read the inline comments below about negative weights, and to try this notebook with and without them.


























# def rmsle(y, y0):
#     assert len(y) == len(y0)
#     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

modelR1 = Ridge(solver="sag", fit_intercept=True, random_state=2, alpha=4, 
                tol=0.0006,
                max_iter=800)
modelR1.fit(X, y)
print('[{}] Finished to train ridge sag'.format(time.time() - start_time))
predsR = modelR1.predict(X=X_test)
print('[{}] Finished to predict ridge sag'.format(time.time() - start_time))

modelR2 = Ridge(solver="sag", fit_intercept=True, random_state=145, alpha = 0.4)
modelR2.fit(X, y)
print('[{}] Finished to train ridge lsqrt'.format(time.time() - start_time))
predsR2 = modelR2.predict(X=X_test)
print('[{}] Finished to predict ridge lsqrt'.format(time.time() - start_time))

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.01, random_state = 144) 
d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192) 
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192) 
watchlist = [d_train, d_valid]

params = {
    'learning_rate': 0.56,
    'application': 'regression',
    'max_depth': 5,
    'num_leaves': 40,
    'verbosity': -1,
    'metric': 'RMSE',
    'feature_fraction':0.8, #cambiato da 0.6
    'nthread': 4
}

params2 = {
    'learning_rate': 0.85,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 110,
    'verbosity': -1,
    'metric': 'RMSE',
    'nthread': 4
}

modelL1 = lgb.train(params, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, \
early_stopping_rounds=None, verbose_eval=500) 
predsL = modelL1.predict(X_test)

print('[{}] Finished to predict lgb 1'.format(time.time() - start_time))

train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.01, random_state = 101) 
d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
watchlist2 = [d_train2, d_valid2]

modelL2 = lgb.train(params2, train_set=d_train2, num_boost_round=3200, valid_sets=watchlist2, \
early_stopping_rounds=None, verbose_eval=500) 
predsL2 = modelL2.predict(X_test)

print('[{}] Finished to predict lgb 2'.format(time.time() - start_time))

preds = predsR2*0.2 + predsR*0.1 + predsL*0.5 + predsL2*0.2

submission['price'] = np.expm1(preds)
submission.to_csv("submission_lgbm_ridge_8.csv", index=False)
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
