#07/02 09:41
import gc
import time
import numpy as np
import pandas as pd
import os            #to get pid : os.getpid()
import psutil        #to get process : psutil.Process(os.getpid()) and memory
import sys           #to get memory usage of an object sys.getsizeof(object)

from scipy.sparse import csr_matrix, hstack, vstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
#WordBatch
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from nltk.corpus import stopwords
import re

sklearn.set_config(True) #faster option

print(">>> changed: preds = np.append(preds, predsFM*0.7 + predsL*0.3)")
print(">>> changed: iters=20")
print(">>> changed: wb_name instead of cv_name")
print(">>> changed: float32 on LGBM dataset")
print(">>> changed: three different cv for category")
print(">>> changed: removed validation on LGBM")
print(">>> changed: 6000 rounds on LGBM")
print(">>> changed: NUM_BRANDS = 3500")
print(">>> added:   hash_polys_maxdf: 0.1")
#TODO -1 come valore della colonna 'missing'

NUM_BRANDS = 3500 #ff1312 4000 < 3000 > 2500
NUM_CATEGORIES = 1200
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 40000

process = psutil.Process(os.getpid())
PAGESIZE = os.sysconf("SC_PAGE_SIZE")

def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")

def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

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

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english') + ['ask', 'size', 'make']}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])
    
def main():
    
    start_time = time.time()
    
    train = pd.read_table('../input/train.tsv', engine='c')
    
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    train = train.drop(train[(train.price < 1.0)].index)

    
    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    print("Memory Usage after reading Train data=", process.memory_info().vms/1000000./PAGESIZE, "Gb")
    
    def initial_cleaning(merge):
        merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
            zip(*merge['category_name'].apply(lambda x: split_cat(x)))
        merge.drop('category_name', axis=1, inplace=True)
        print('[{}] Split categories completed.'.format(time.time() - start_time))

        handle_missing_inplace(merge)
        print('[{}] Finished to handle missing'.format(time.time() - start_time))
    
        cutting(merge)
        print('[{}] Finished to cut'.format(time.time() - start_time))
    
        to_categorical(merge)
        print('[{}] Finished to convert categorical'.format(time.time() - start_time))
    
#        merge.item_description = merge.item_description.str.lower()
#        normalize_desc(merge)
#        print('[{}] Finished to normalize'.format(time.time() - start_time))
        
        merge['shipping'] = merge['shipping'].astype(str)
        merge['item_condition_id'] = merge['item_condition_id'].astype(str)
        print("Memory Usage after Normalizing=", process.memory_info().vms/1000000./PAGESIZE, "Gb")
    
    #%%
    initial_cleaning(train)
    
    print('[{}] Starting CV'.format(time.time() - start_time))
    cv_name = CountVectorizer(min_df=NAME_MIN_DF, ngram_range=(1, 2), binary=True)
    cv_cat1 = CountVectorizer(dtype='int32', binary=True)
    cv_cat2 = CountVectorizer(dtype='int32', binary=True)
    cv_cat3 = CountVectorizer(dtype='int32', binary=True)
    cv_brand = CountVectorizer(dtype='int32', binary=True)
    cv_ship = CountVectorizer(token_pattern='\d+', dtype='int32', lowercase=False)
    cv_cond = CountVectorizer(token_pattern='\d+', dtype='int32', lowercase=False)
    wb_desc = wordbatch.WordBatch(normalize_text, 
                             extractor=(WordBag, {"hash_ngrams": 2, 
                                                  "hash_ngrams_weights": [1.0, 1.0],
                                                  "hash_size": 2 ** 28, 
                                                  "hash_polys_maxdf": 0.1,
                                                  "norm": "l2", 
                                                  "tf": 1.0,
                                                  "idf": None}),
                             procs=4)
    wb_desc.dictionary_freeze= True
    wb_name = wordbatch.WordBatch(normalize_text, 
                                extractor=(WordBag, {"hash_ngrams": 2, 
                                    "hash_ngrams_weights": [1.5, 1.0],
                                    "hash_size": 2 ** 26, 
                                    "hash_polys_maxdf": 0.1,
                                    "norm": None, 
                                    "tf": 'binary',
                                    "idf": None}), 
                                procs=4)
    wb_name.dictionary_freeze= True
    
    fu = FeatureUnion(
            transformer_list=[
                # Pipeline for 'name' CountVectorizer()
#                ('cv_name', Pipeline([
#                    ('name', ItemSelector(key='name')),
#                    ('cv', cv_name)
#                ])),
                ('wb_name', Pipeline([
                    ('name', ItemSelector(key='name')),
                    ('wb', wb_name)
                ])),
                ('cv_cat1', Pipeline([
                    ('cat1', ItemSelector(key='general_cat')),
                    ('cv', cv_cat1)
                ])),
                ('cv_cat2', Pipeline([
                    ('cat2', ItemSelector(key='subcat_1')),
                    ('cv', cv_cat2)
                ])),
                ('cv_cat3', Pipeline([
                    ('cat3', ItemSelector(key='subcat_2')),
                    ('cv', cv_cat3)
                ])),
                ('cv_brand', Pipeline([
                    ('brand', ItemSelector(key='brand_name')),
                    ('cv', cv_brand)
                ])),
                ('cv_ship', Pipeline([
                    ('ship', ItemSelector(key='shipping')),
                    ('cv', cv_ship)
                ])),
                ('cv_cond', Pipeline([
                    ('cond', ItemSelector(key='item_condition_id')),
                    ('cv', cv_cond)
                ])),
                ('wb_desc', Pipeline([
                    ('desc', ItemSelector(key='item_description')),
                    ('wb', wb_desc)
                ]))
    
            ], 
            n_jobs=1 #Must be 1 otherwise WordBatch fails
            )
                
    X_fu = fu.fit_transform(train)
    
    print('[{}] Finished  vectorizing `ALL`'.format(time.time() - start_time), X_fu.shape)
    print("Memory Usage after first FE=", process.memory_info().vms/1000000./PAGESIZE, "Gb")
    
    X_fu_mask = np.where(X_fu.getnnz(axis=0) > 1)[0]
#    X_fu_mask = np.array(np.clip(X_fu.getnnz(axis=0) - 1, 0, 1), dtype=bool)
    X_fu = X_fu[:, X_fu_mask]
    print('[{}] Finished  removing infrequent columns'.format(time.time() - start_time), X_fu.shape)
    
    from nltk.corpus import stopwords
    interesting_words = ['new', 'perfect', 'fit', 'used', #'super', 'cute', 'excellent',
                     'great', 'retail', '[rm]', 'never used', 'bundle', #'diamond', 'ruby',
                 'platinum', 'gold', 'set', 'case', 'unused', 'unopened', 'sealed',
             'with authenticity', 'china glaze', 'authenticity card', 'complete series', #new
             'camera bag', 'no box', 'original receipt', 'day usps', 'not pictured',
             'serious offers', 'box factory' ]
    X_intcol = pd.DataFrame()
    X_intcol_test = pd.DataFrame()
    for word in interesting_words:
        X_intcol[word] = train['item_description'].apply(lambda x : word in x)
    
    X_des = train['item_description'].apply(lambda x: len(x)).astype('float32')
    X_des = X_des[:, np.newaxis]
    scaler = MaxAbsScaler()
    X_des = scaler.fit_transform(X_des)
    
    
    #ignore_words = ['cant', 'ask', 'size', 'inch', 'inches', 'already', 'inside', 'easy']
    #stop = stopwords.words('english') + ignore_words
    #FF
    #FF version 5 n_features=2**18 --> n_features=2**17
    #hv = HashingVectorizer(input='content', stop_words= stop, n_features=2**17,
    #                        lowercase=False,
    #                        decode_error='ignore')
    #X_description = hv.transform(merge['item_description'])
    #print('[{}] Finished Hash vectorize `item_description`'.format(time.time() - start_time))
    
    sparse_merge = hstack(( X_fu, X_intcol, X_des)).tocsr()
    
    print('[{}] Finished to create TRAIN sparse merge'.format(time.time() - start_time), 
          sparse_merge.shape)
    
    X = sparse_merge #[:nrow_train]
    
    print("Memory Usage before training=", process.memory_info().vms/1000000./PAGESIZE, "Gb")
    
    
    modelFM = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=220, e_noise=0.0001, iters=20, inv_link="identity", threads=4)

    modelFM.fit(X, y, verbose=1)
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))


    X_mask = np.where(X.getnnz(axis=0) > 100)[0]
#    X_mask = np.array(np.clip(X.getnnz(axis=0) - 100, 0, 1), dtype=bool)
    X = X[:, X_mask].astype('float32')
    print('[{}] Finished  removing infrequent columns for lgbm'.format(time.time() - start_time), X.shape)
    
    d_train = lgb.Dataset(X, label=y) 
    watchlist = [d_train]

    params = {
        'learning_rate': 0.58,
        'application': 'regression',
        'max_depth': 6,
        'num_leaves': 48,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.6,
        'nthread': 4,
        'min_data_in_leaf': 40,
        'max_bin': 32
    }
    
    modelL1 = lgb.train(params, train_set=d_train, num_boost_round=5000, #valid_sets=watchlist, \
                        early_stopping_rounds=None, verbose_eval=False) 
    
    print('[{}] Finished to train lgb 1'.format(time.time() - start_time))

    del train, d_train
    gc.collect()
    print("Memory Usage before predicting=", process.memory_info().vms/1000000./PAGESIZE, "Gb")

    #START PREDICTING Test in batches
    MAX_BATCH = 350000
    submission = pd.DataFrame()
    preds = np.array([])
    try:
        for df in pd.read_table('../input/test.tsv', engine='c', 
                                   chunksize = MAX_BATCH):
            print(df.shape)
            submission = submission.append(df[['test_id']])
            initial_cleaning(df)
            X_fu_test = fu.transform(df)
            X_fu_test = X_fu_test[:, X_fu_mask]
            X_intcol_test = pd.DataFrame()
            for word in interesting_words:
                X_intcol_test[word] = df['item_description'].apply(lambda x : word in x)
            X_des_test = df['item_description'].apply(lambda x: len(x)).astype('float32')
            X_des_test = X_des_test[:, np.newaxis]
            X_des_test = scaler.transform(X_des_test)
            sparse_merge_test = hstack(( X_fu_test, X_intcol_test, X_des_test)).tocsr()
            print('[{}] Finished to create batch sparse merge test'.format(time.time() - start_time),
                  sparse_merge_test.shape)
            X_test = sparse_merge_test
            predsFM = modelFM.predict(X=X_test)
            print('[{}] Finished to predict first model '.format(time.time() - start_time))
            X_test = X_test[:, X_mask].astype('float32')
            print('[{}] Finished  removing columns'.format(time.time() - start_time), X_test.shape)
            predsL = modelL1.predict(X_test)
            print('[{}] Finished to predict lgbm'.format(time.time() - start_time))
            preds = np.append(preds, predsFM*0.71 + predsL*0.29)
    except:
        raise
#    preds = preds[1:] #ff Remove the first null value
    print(len(preds), submission.shape )
    
    print("Memory Usage before submitting=", process.memory_info().vms/1000000./PAGESIZE, "Gb")

    submission['price'] = np.expm1(preds)
    submission.to_csv("fu_batch350.csv", index=False)

if __name__ == '__main__':
    main()