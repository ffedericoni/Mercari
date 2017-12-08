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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

NUM_BRANDS = 4000
#NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 25000
CELL=False

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

#just commented this
#handle_no_description(merge)
#print('[{}] Finished to copy names to missing desc'.format(time.time() - start_time))

cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))

cv = CountVectorizer(dtype='int32') #FF added dtype=...
X_category = cv.fit_transform(merge['category_name'])
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))
#%%
#FF START
if CELL:
    start_time = time.time()
from nltk.corpus import stopwords
interesting_words = ['new', 'perfect', 'fit', 'used', 'super', 'cute', 'excellent',
                 'great', 'retail', '[rm]', 'never', 'bundle', 'diamond', 'ruby',
                 'platinum', 'gold', 'set', 'case']
X_intcol = pd.DataFrame()
for word in interesting_words:
    X_intcol[word] = merge['item_description'].apply(lambda x : word in x.lower())

X_des = merge['item_description'].apply(lambda x: len(x)).astype('float32')
X_des = X_des[:, np.newaxis]
scaler = MinMaxScaler()
X_des = scaler.fit_transform(X_des)

ignore_words = ['cant', 'ask', 'size', 'inch', 'inches', 'already', 'inside', 'easy']
stop = stopwords.words('english') + ignore_words #+ interesting_words 
#FF END

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                     ngram_range=(1, 3),
                     stop_words=stop)
X_description = tv.fit_transform(merge['item_description'])
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))
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
#%%
#==============================================================================
#if CELL:
#    start_time = time.time()
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler(with_mean=False)
#scaler.fit_transform(sparse_merge)
#print('[{}] Time taken to scale'.format(time.time() - start_time))
#==============================================================================
#%%
if CELL:
    start_time = time.time()
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

# def rmsle(y, y0):
#     assert len(y) == len(y0)
#     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

#TODO experiment with tol=0.01
modelR1 = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)
modelR1.fit(X, y)
print('[{}] Finished to train ridge sag'.format(time.time() - start_time))
predsR = modelR1.predict(X=X_test)
print('[{}] Finished to predict ridge sag'.format(time.time() - start_time))

modelR2 = Ridge(solver="sag", fit_intercept=True, random_state=145, alpha = 1.0)
modelR2.fit(X, y)
print('[{}] Finished to train ridge lsqrt'.format(time.time() - start_time))
predsR2 = modelR2.predict(X=X_test)
print('[{}] Finished to predict ridge lsqrt'.format(time.time() - start_time))
#%%
if CELL:
    start_time = time.time()
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.01, random_state = 144) 
d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
watchlist = [d_train, d_valid]

params = {
    'learning_rate': 0.76,
    'application': 'regression',
    'max_depth': 5,
    'num_leaves': 40,
    'verbosity': -1,
    'metric': 'RMSE',
    'feature_fraction':0.6, #cambiato da 0.7
    'nthread': 3
}

params2 = {
    'learning_rate': 1.0,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 80,
    'verbosity': -1,
    'metric': 'RMSE',
    'bagging_fraction':0.8, # aggiunto ora
    'nthread': 3
}

modelL1 = lgb.train(params, train_set=d_train, num_boost_round=2500, valid_sets=watchlist, \
early_stopping_rounds=None, verbose_eval=500) 
predsL = modelL1.predict(X_test)

print('[{}] Finished to predict lgb 1'.format(time.time() - start_time))
#%%
if CELL:
    start_time = time.time()
train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.01, random_state = 101) 
d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
watchlist2 = [d_train2, d_valid2]

modelL2 = lgb.train(params2, train_set=d_train2, num_boost_round=1500, valid_sets=watchlist2, \
early_stopping_rounds=None, verbose_eval=500) 
predsL2 = modelL2.predict(X_test)

print('[{}] Finished to predict lgb 2'.format(time.time() - start_time))

preds = predsR2*0.15 + predsR*0.05 + predsL*0.6 + predsL2*0.2

submission['price'] = np.expm1(preds)
#submission.to_csv("submission_lgbm_ridge_8.csv", index=False)
#TODO cerca elementi con maggiore errore
#TODO aggiungi al Kernel la colonna des_len 
#TODO aggiorna sul Kernel le parole interessanti e non
#TODO dai peso maggiore alle colonne delle parole interessanti
#TODO rimuovi prezzi a 0.0 o inserisci prezzi medi per categoria?
#TODO Parallelize the Vectorizations using the 4 cores
#TODO GaussianNB partial_fit to manage batches of data
#TODO check feature importances
#==============================================================================
# for ind in range(10):
#     print("Truth=", y[ind], 
#     "L1=", modelL1.predict(X[ind]),
#     "L2=", modelL2.predict(X[ind]),
#     "R1=", modelR1.predict(X[ind]),
#     "R2=", modelR2.predict(X[ind]))
#==============================================================================
#%%
from sklearn.metrics import mean_squared_error
predTL1 = modelL1.predict(X)
predTL2 = modelL2.predict(X)
predTR1 = modelR1.predict(X)
predTR2 = modelR2.predict(X)
predall = predTL1*0.5 + predTL2*0.2 + predTR1*0.15 + predTR2*0.15
print("MSE:", 
    "L1=", mean_squared_error(y,predTL1),
    "L2=", mean_squared_error(y,predTL2),
    "R1=", mean_squared_error(y,predTR1),
    "R2=", mean_squared_error(y,predTR2),
    "\nALL=", mean_squared_error(y,predall)
    )

high_err = merge.loc[(y-predall) > 2.0]
low_err = merge.loc[(y-predall) < -2.0]

print("Much Higher priced items: ", len(high_err))
print("Much Lower priced items: ", len(low_err))

#for ind in range( 1000 ):
#    if  y[ind] - predall[ind] > 1.0:
#        print("----", merge.iloc[ind, 5], "$$", merge.iloc[ind, 4], "###",\
#         merge.iloc[ind, 3])

#==============================================================================
#predall = predTL1*0.65 + predTL2*0.05 + predTR1*0.15 + predTR2*0.15
#MSE: L1= 0.18086623326 L2= 0.206544236825 R1= 0.205544041285 R2= 0.205058258114 
#ALL= 0.17742740028354798
#Much Higher priced items:  1942
#Much Lower priced items:  1381
#==============================================================================

