import gc
import time
import numpy as np
import pandas as pd
import os            #to get pid : os.getpid()
import psutil        #to get process : psutil.Process(os.getpid()) and memory
import sys           #to get memory usage of an object sys.getsizeof(object)

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import sklearn

sklearn.set_config(True) #faster option

NUM_BRANDS = 3000
#NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 40000
CELL=True

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
#16g is also 16 grams    dataset.item_description = dataset.item_description.str.replace(' 16g ', ' 16gb ')

def sparse_df_size_in_mb(sparse_df):
    '''
    Size of a sparse matrix in Mbytes
    '''
    size_in_bytes = sparse_df.data.nbytes
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_mb



start_time = time.time()
process = psutil.Process(os.getpid())
PAGESIZE = os.sysconf("SC_PAGE_SIZE")

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')
print('[{}] Finished to load data'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

nrow_train = train.shape[0]
y = np.log1p(train["price"]) #this is because objective is RMSE, but should be RMSLE
merge: pd.DataFrame = pd.concat([train, test])
submission: pd.DataFrame = test[['test_id']]


del train
del test
gc.collect()
print("Memory Usage after Garbage Collection=", process.memory_info().vms/1000000./PAGESIZE, "Gb")

#%%
handle_missing_inplace(merge)
print('[{}] Finished to handle missing'.format(time.time() - start_time))

cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))

merge.item_description = merge.item_description.str.lower()

normalize_desc(merge)
print('[{}] Finished to normalize'.format(time.time() - start_time))


#just commented this
#handle_no_description(merge)
#print('[{}] Finished to copy names to missing desc'.format(time.time() - start_time))

cv = CountVectorizer(min_df=NAME_MIN_DF, dtype='int32')
X_name = cv.fit_transform(merge['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer(dtype='int32') #FF added dtype=...
X_category = cv.fit_transform(merge['category_name'])
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))
#%%
#FF START
if CELL:
    start_time = time.time()

interesting_words = ['new', 'perfect', 'fit', 'used', 
#                     'super', 'cute', 'excellent',
                 'great', 'retail', '[rm]', 'never used', 'bundle', 
#                 'diamond', 'ruby', 'platinum',
                 'gold', 'set', 'case', 'unused', 'unopened', 'sealed']
X_intcol = pd.DataFrame()
for word in interesting_words:
    X_intcol[word] = merge['item_description'].apply(lambda x : word in x.lower())

X_des = merge['item_description'].apply(lambda x: len(x)).astype('float32')
X_des = X_des[:, np.newaxis]
scaler = MaxAbsScaler()
X_des = scaler.fit_transform(X_des)

ignore_words = ['cant', 'ask', 'size', 'inch', 'inches', 'already', 'inside', 
                'easy'
                ]
stop = stopwords.words('english') + ignore_words #+ interesting_words 
#FF END
from sklearn.feature_extraction.text  import HashingVectorizer
hv = HashingVectorizer(input='content', stop_words= stop, n_features=2**17,
                       dtype='float32',
                       binary=False)
X_description = hv.transform(merge['item_description'])

#==============================================================================
# tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
#                      ngram_range=(1, 3),
#                      stop_words=stop)
# X_description = tv.fit_transform(merge['item_description'])
#==============================================================================
print('[{}] Finished Hash vectorize `item_description`'.format(time.time() - start_time))

#==============================================================================
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
# X_SVD_des = lsa.fit_transform(X_description)
#==============================================================================

#%%
if CELL:
    start_time = time.time()
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, 
                        X_name, X_intcol, X_des)).tocsr()
gc.collect()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))
print("Memory Usage=", process.memory_info().vms/1000000./PAGESIZE, "Gb")


X_train = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

#%%

start_time = time.time()

lgbmodel = lgb.LGBMRegressor(
#    learning_rate= 0.76,
    objective='regression',
    min_child_weight=0.001, #default=0.001
    min_data_in_leaf=100,
    bagging_fraction = 1,
    silent=True,
    metric='rmse',
#    train_metric=False,
#    metric_freq=10,
    n_estimators=7000,
#    cat_smooth=10, #this can reduce the effect of noises in categorical features, especially for categories with few data
    max_bin=8192, #TODO try to reduce
#    boosting='dart',
    num_threads=4
    )

param_grid = {
    'learning_rate': [0.66, 0.71],
    'max_depth': [5],
    'num_leaves': [40, 60],
    'feature_fraction': [0.8],
#    'cat_smooth': [10],
#    'min_data_in_leaf': [20],
#    'min_split_gain': [0.0],
#    'bagging_freq':[1, 2, 4],
#    'bagging_fraction':[0.6, 0.8]

#    'max_bin':8192
}
#best is 
gbm = GridSearchCV(lgbmodel, param_grid, cv=5,
                   verbose=10, 
#                   scoring='neg_mean_absolute_error',
                   n_jobs=1)

gbm.fit(X_train, y)
print('[{}] Finished to GridSearch'.format(time.time() - start_time))
print('Best parameters found by grid search are:', gbm.best_params_)
#merge[merge.item_description.str.contains(' 64g ')]
#merge.item_description.str.contains(' iphone 5 ').sum()
