# coding: utf-8

# # Mercari Price Suggestion Challenge
# 
#TODO LabelEncoding
#use TextBlob to identify adjectives and build a column with the encoded adjectives
#JJ adjective: nice, easy 
#JJR adjective, comparative: nicer, easier
#JJS adjective, superlative: nicest, easiest 
#from pattern.en import number
#number(string)    # "seventy-five point two" => 75.2
#TODO filter out "No description yet" in Item Description
#TODO build an external file with adjectives and look for adjectives used on higher prices under same category
#TODO replace xgboost with lightgbm
# In[ ]:


import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
#import urllib        #for url stuff
#import re            #for processing regular expressions
#import datetime      #for datetime operations
#import calendar      #for calendar for datetime operations
import time          #to get the system time
import sys           #to get memory usage of an object
import resource      #to get memory usage of the process
import psutil
#import scipy         #for other dependancies
import seaborn as sns #for making plots
import matplotlib.pyplot as plt # for plotting
import os                # for os commands
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import LabelBinarizer
import lightgbm as lgb
import gc

KERNEL=False
process = psutil.Process(os.getpid())
# ## Reading in the files..
# In[ ]:

start = fixstart = time.time()
train_col_types = {'train_id':'int32', 'name':'str', 'item_condition_id':'int8', 
                   'category_name':'str', 'brand_name':'str',
                   'price':'float32', 'shipping':'bool', 'item_description':'str'}
test_col_types = {'train_id':'int32', 'name':'str', 'item_condition_id':'int8', 
                   'category_name':'str', 'brand_name':'str',
                   'shipping':'bool', 'item_description':'str'}
train_df = pd.read_csv('../input/train.tsv', sep='\t', na_filter = True,
                       dtype=train_col_types )
test_df = pd.read_csv('../input/test.tsv', sep='\t', na_filter = True,
                       dtype=test_col_types )
end = time.time()
print("Time taken reading input files {}.".format((end-start)))
# In[ ]:

# # Basic feature engineering 

# new features
# len of description
if not KERNEL:
    start = time.time()

train_df['item_description'].fillna(value='UNKNOWN', inplace=True)
test_df['item_description'].fillna(value='UNKNOWN', inplace=True)

train_df['des_len'] = train_df['item_description'].apply(lambda x: len(x)).astype('int16')
test_df['des_len'] = test_df['item_description'].apply(lambda x: len(x)).astype('int16')

# words in description
train_df['word_count'] = train_df['item_description'].apply(lambda x: len(x.split())).astype('int16')
test_df['word_count'] = test_df['item_description'].apply(lambda x: len(x.split())).astype('int16')

# description contains interesting words
interesting_words = ['new', 'perfect', 'fit', 'used', 'super', 'cute', 'excellent',
                     'great', 'retail', '[rm]', 'never' ]
for word in interesting_words:
    train_df[word] = train_df['item_description'].apply(lambda x : word in x.lower())
    test_df[word] = test_df['item_description'].apply(lambda x : word in x.lower())

if not KERNEL:
    end = time.time()
    print("Time taken with words feature engineering {}.".format((end-start)))

# - **1. Category label features - **

# In[ ]:


# 1. Extract 3 category related features 
def cat_split(row):
    try:
        text = row
        txt1, txt2, txt3 = text.split('/')
        return txt1, txt2, txt3
    except:
        return np.nan, np.nan, np.nan


train_df["cat_1"], train_df["cat_2"], train_df["cat_3"] = zip(*train_df.category_name.apply(lambda val: cat_split(val)))
test_df["cat_1"], test_df["cat_2"], test_df["cat_3"] = zip(*test_df.category_name.apply(lambda val: cat_split(val)))



# ** 2. if category present -yes/no features -**

# In[ ]:


train_df['if_cat'] = train_df.category_name.apply(lambda row : row == row)
test_df['if_cat'] = test_df.category_name.apply(lambda row : row == row)


# **3. If brand name is present - yes/no features -**

# In[ ]:

train_df['if_brand'] = train_df.brand_name.apply(lambda row : row == row)
test_df['if_brand'] = test_df.brand_name.apply(lambda row : row == row)



# In[ ]:
from sklearn.preprocessing import LabelEncoder

def CatEncoder(train_df, test_df, col_name):
    le = LabelEncoder()
    train_df[col_name].fillna('UNK', inplace=True)
    test_df[col_name].fillna('UNK', inplace=True)
    le = le.fit(train_df[col_name].tolist() + test_df[col_name].tolist())
    train_df[col_name] = le.transform(train_df[col_name])
    test_df[col_name] = le.transform(test_df[col_name])
    train_df[col_name] = train_df[col_name].astype('int32')
    train_df[col_name] = train_df[col_name].astype('int32')
    return train_df, test_df

train_df, test_df = CatEncoder(train_df, test_df, 'cat_1')
train_df, test_df = CatEncoder(train_df, test_df, 'cat_2')
train_df, test_df = CatEncoder(train_df, test_df, 'cat_3')
train_df, test_df = CatEncoder(train_df, test_df, 'brand_name')

# ** 5. if item_description present - yes/no feature -**

# In[ ]:


# item description related features 
print("Description of item is not present in {}".format(train_df.loc[train_df.item_description == 'No description yet'].shape[0]))
print("while the shape of train_df is {}".format(train_df.shape[0]))

train_df['if_description'] = train_df.item_description.apply(lambda row : row == 'No description yet')
test_df['if_description'] = test_df.item_description.apply(lambda row : row == 'No description yet')


#%%
#Create column with only adjectives in Item Description
def extract_adjs(row):
    '''function to return the adjectives in the text'''
    adjs = ''
    tokens = nltk.word_tokenize(row)
    tagged_tokens = nltk.pos_tag(tokens)
    for co in tagged_tokens:
        if co[1] == 'JJ':
            adjs = adjs + co[0] + ' '
    return adjs    


if not KERNEL:
    print("Memory Usage before Vectorizing:", process.memory_info().rss)

# ** 6. SVD on tf-idf on unigrams for item_description -**

# In[ ]:


# description related tf-idf features 
# I guess "No dscription present won't affact these features ... So, I am not removing them.
import time
start = time.time()
stop = stopwords.words('english') + interesting_words + ['cant', 'ask', 'size']
tfidf_vec = TfidfVectorizer(stop_words=stop, min_df = 1.0/100000.0, ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['item_description'].values.tolist() + test_df['item_description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['item_description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['item_description'].values.tolist())
#%%
n_comp = 16
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
end = time.time()
print("time taken Vectorizing and Decomposing Item Description: {}".format(end - start))
#%%
print(train_df.shape[0])
train_df = train_df.loc[train_df.item_description == train_df.item_description]
test_df = test_df.loc[test_df.item_description == test_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]
test_df = test_df.loc[test_df.name == test_df.name]
print(train_df.shape[0])
print("Dropped records where item description was nan")


# ** 7. SVD on tf-idf of unigram of product name features - **

# In[ ]:


# product name related features 
start = time.time()
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['name'].values.tolist() + test_df['name'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['name'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['name'].values.tolist())

n_comp = 16 #should be 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
end = time.time()
print("time taken Vectorizing and Decomposing Product Name: {}".format(end - start))
if not KERNEL:
    print("Memory Usage after Vectorizing:", process.memory_info().rss)

# In[ ]:


# test check for dimensions before model 
print("Train should have one columns more than test")
print(train_df.shape[1])
print(test_df.shape[1])
print("perfect The data is fine")


#  # XGboost regressor ...
#  Now we have 49 features which could be used in price prediction and let's use them and see how they are performing 

# In[ ]:
#Reduce Memory fooprint
col = [c for c in train_df.columns if c.startswith('svd_')]
for c in col:
    train_df[c] = train_df[c].astype('float32')
    test_df[c] = test_df[c].astype('float32')

train_df['item_condition_id'] = train_df['item_condition_id'].astype('int8')
test_df['shipping'] = test_df['shipping'].astype('int8')

if not KERNEL:
    print("Memory Usage after Reducing Memory fooprint:", process.memory_info().rss)


# XGboost regressor ...
# replace all nan with -1 
#print(train_df.isnull().sum())
train_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)
#print(train_df.isnull().sum())


# In[ ]:


train = train_df #.copy()
test = test_df #.copy()
print("Difference of features in train and test are {}".format(np.setdiff1d(train.columns, test.columns)))
print("")
do_not_use_for_training = ['test_id','train_id','name', 'category_name', 'price', 'item_description']
#do_not_use_for_training = ['cat_1','test_id','cat_2','cat_3','train_id','name', 'category_name', 'brand_name', 'price', 'item_description']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
print("We will be using following features for training {}.".format(feature_names))
print("")
print("Total number of features are {}.".format(len(feature_names)))
if not KERNEL:
    print("Memory Usage of Train DataFrame:", sys.getsizeof(train_df))

# In[ ]:


y = np.log(train['price'].values + 1)



# In[ ]:


from sklearn.model_selection import train_test_split
Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, 
                                    test_size=0.1, 
                                    random_state=1986)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(test[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

start = time.time()
if not KERNEL:
    nthread = 3
else:
    nthread = -1
xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 
           'max_depth': 15, 'subsample': 1.0, 'lambda': 2.0, 'nthread': nthread, 
	   'booster' : 'gbtree', 'silent': 1, 'max_delta_step': 10,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
ntree = 250
model_1 = xgb.train(xgb_par, dtrain, ntree, watchlist, early_stopping_rounds=20, 
                    maximize=False, verbose_eval=10)
print('Modeling RMSLE %.5f' % model_1.best_score)
end = time.time()
print("Time taken in training is {}.".format(end - start))
print("Parameters: ", xgb_par, ntree)

# In[ ]:
print("Memory Usage after Training:", process.memory_info().rss)


if not KERNEL:
    start = time.time()
yvalid = model_1.predict(dvalid)
ytest = model_1.predict(dtest)
if not KERNEL:
    end = time.time()
    print("Time taken in prediction is {}.".format(end - start))

# In[ ]:


start = time.time()
if test.shape[0] == ytest.shape[0]:
    print('Test shape OK.') 
test['price'] = np.exp(ytest) - 1
test[['test_id', 'price']].to_csv('ff_xgb_mercari.csv', index=False)
end = time.time()
print("Time taken in writing file is {}.".format(end - start))
print("Time taken by Kernel {}.".format(end - fixstart))

# **Sleep time zzzzz......**
# # Upvote if you find this analysis useful... Thanks (y)