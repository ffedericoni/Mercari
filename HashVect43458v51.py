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
from sklearn.model_selection import train_test_split, cross_val_score
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


def main():
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

#    modelR2 = Ridge(solver="sag", fit_intercept=True, random_state=145, alpha = 0.4)
#    modelR2.fit(X, y)
#    print('[{}] Finished to train ridge lsqrt'.format(time.time() - start_time))
#    predsR2 = modelR2.predict(X=X_test)
#    print('[{}] Finished to predict ridge lsqrt'.format(time.time() - start_time))

    modelR3 = Ridge(solver="saga", fit_intercept=False, random_state=101, 
                    copy_X=True,
                    max_iter=200,
                    tol=0.005,
                    alpha = 0.5)
    scaler = MaxAbsScaler()
    X_scale = scaler.fit_transform(X)
    X_test_scale = scaler.transform(X_test)
    modelR3.fit(X_scale, y)
    print('[{}] Finished to train ridge SAGA'.format(time.time() - start_time))
    predsR3 = modelR3.predict(X=X_test_scale)
    print('[{}] Finished to predict ridge SAGA'.format(time.time() - start_time))

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

    modelL1 = lgb.train(params, train_set=d_train, num_boost_round=8900, valid_sets=watchlist, \
    early_stopping_rounds=None, verbose_eval=500) 
    predsL = modelL1.predict(X_test)
    
    print('[{}] Finished to predict lgb 1'.format(time.time() - start_time))
    
    train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.01, random_state = 101) 
    d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
    d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
    watchlist2 = [d_train2, d_valid2]

    modelL2 = lgb.train(params2, train_set=d_train2, num_boost_round=3500, valid_sets=watchlist2, \
    early_stopping_rounds=None, verbose_eval=500) 
    predsL2 = modelL2.predict(X_test)

    print('[{}] Finished to predict lgb 2'.format(time.time() - start_time))

    preds = predsR*0.1 + predsL*0.6 + predsL2*0.2 + predsR3*0.1 #predsR2*0.20 + 

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


if __name__ == '__main__':
    main()