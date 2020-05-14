#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style('dark')


# In[ ]:


MODEL_DIR= 'path/to/saved/model/'
TEST_FILES='path/to/test/files/'
seed=42     # For reproducible results
total_samples = 1000000
num_samples = 200000
folds=[0]


# In[ ]:


column_names=['pt1','pt2','invmass','dphi','met','dphi1_met','dphi2_met']
ttbar = pd.concat([pd.read_csv(FILES_DIR+file,sep=' ', index_col=None, usecols=[1,2,3,4,5,6,7], names=column_names) for file in os.listdir(FILES_DIR) if 'ttbar' in file]).reset_index(drop=True)
sig = pd.concat([pd.read_csv(FILES_DIR+file,sep=' ', index_col=None,usecols=[1,2,3,4,5,6,7], names=column_names) for file in os.listdir(FILES_DIR) if 'sig' in file]).reset_index(drop=True)
test1 = pd.read_csv(TEST_FILES+'data1.txt',sep=' ', index_col=None,usecols=[1,2,3,4,5,6,7], names=column_names).reset_index(drop=True)
test2 = pd.read_csv(TEST_FILES+'data2.txt',sep=' ', index_col=None,usecols=[1,2,3,4,5,6,7], names=column_names).reset_index(drop=True)
test3 = pd.read_csv(TEST_FILES+'data3.txt',sep=' ', index_col=None,usecols=[1,2,3,4,5,6,7], names=column_names).reset_index(drop=True)
bkg3 = pd.read_csv(TEST_FILES+'bkg3.txt',sep=' ', index_col=None,usecols=[1,2,3,4,5,6,7], names=column_names).reset_index(drop=True)
test4 = pd.read_csv(TEST_FILES+'data4.txt',sep=' ', index_col=None,usecols=[1,2,3,4,5,6,7], names=column_names).reset_index(drop=True)


# In[ ]:


ttbar['dphi']=abs(ttbar['dphi'])
test1['dphi']=abs(test1['dphi'])
test2['dphi']=abs(test2['dphi'])
test3['dphi']=abs(test3['dphi'])
bkg3['dphi']=abs(bkg3['dphi'])
test4['dphi']=abs(test4['dphi'])


# In[ ]:


def dist_plot_test(feat, data):
#     plt.figure(figsize=(8,8))
    ax=sns.distplot(data[feat], norm_hist=True)
    plt.legend()
    plt.xlabel(feat, fontsize=20)
    plt.ylabel('Normalized', fontsize=20)
    plt.title(f'Distribution of {feat}', fontsize=20)


# In[ ]:


plt.figure(figsize=(16, 30))

for i in np.arange(0,5,2):
    plt.subplot(4,2,i+1)
    dist_plot_test(column_names[i], test2)
    plt.subplot(4,2,i+2)
    dist_plot_test(column_names[i+1], test2)
    
plt.subplot(4,2,7)
dist_plot_test(column_names[-1], test2)
plt.tight_layout()

plt.savefig(f'plot1.png')


# In[ ]:


traindf = data[data['fold'].isin(folds)].reset_index(drop=True)
tt=traindf[traindf['label']==0].sample(n=10000, random_state=seed).reset_index(drop=True)
ss=traindf[traindf['label']==1].sample(n=10000, random_state=seed).reset_index(drop=True)
traindf=pd.concat([tt,ss])
trn_X = traindf.drop(['label','fold'],1)

# preprocessing done. No effect on BDT. Must for NN. 
mean, std = trn_X.mean(axis=0), trn_X.std(axis=0)
trn_X = (trn_X - mean)/std
trn_X = trn_X.values
trn_Y = traindf['label'].values
valdf =  data[~data['fold'].isin(folds)].sample(n=100000, random_state=seed).reset_index(drop=True)
# valdf = data[~data['fold'].isin(folds)].reset_index(drop=True)

val_X = valdf.drop(['label','fold'],1)

val_X = (val_X - mean)/std   # Validation data statistics not used. Training statistics itself used.
val_X = val_X.values
val_Y = valdf['label'].values


# preprocess test data

test1 = (test1-mean)/std
test1_X = test1.values

test2 = (test2-mean)/std
test2_X = test2.values

test3 = (test3-mean)/std
test3_X = test3.values

bkg3 = (bkg3-mean)/std
bkg3_X = bkg3.values

test4 = (test4-mean)/std
test4_X = test4.values


# In[ ]:


print('Train shape: ', len(trn_X))
print('Validation shape: ', len(val_X))
print('Test1 shape:', len(test1_X))
print('Test2 shape:', len(test2_X))
print('Test3 shape:', len(test3_X))
print('BKG3 shape:', len(bkg3_X))
print('Test4 shape:', len(test4_X))


# In[ ]:


t_df = pd.DataFrame()
v_df = pd.DataFrame()
test_df = pd.DataFrame()

t_df['train_truth'] = trn_Y
t_df['train_predicted'] = 0
t_df['train_pred_proba'] = 0
v_df['val_truth'] = val_Y
v_df['val_predicted'] = 0
v_df['val_pred_proba'] = 0
feat_impdf = pd.DataFrame()
feat_impdf['features'] = column_names

test_df['test1_pred_proba'] = 0
test_df['test2_pred_proba'] = 0

model = XGBClassifier(random_state=seed, n_estimators=500)
model.fit(trn_X, trn_Y, eval_set=[(val_X, val_Y)], eval_metric=['auc'], verbose=10, early_stopping_rounds=30)
val_pred_proba = model.predict_proba(val_X)[:,1] # predicts probability of the event being Signal
train_pred_proba = model.predict_proba(trn_X)[:,1]

test_df['test1_pred_proba'] = model.predict_proba(test1_X)[:,1]
test_df['test2_pred_proba'] = model.predict_proba(test2_X)[:,1]

train_score = roc_auc_score(trn_Y, train_pred_proba)
val_score = roc_auc_score(val_Y, val_pred_proba)
print('Train ROC-AUC: ', np.round(train_score,4))
print('Validation ROC-AUC: ', np.round(val_score,4))
t_df['train_pred_proba']=train_pred_proba
v_df['val_pred_proba']=val_pred_proba
t_df['train_predicted']=model.predict(trn_X)
v_df['val_predicted']=model.predict(val_X)
train_score = accuracy_score(trn_Y, t_df['train_predicted'].values)
val_score = accuracy_score(val_Y, v_df['val_predicted'].values)
print('Train Accuracy: ', np.round(train_score,4))
print('Validation Accuracy: ', np.round(val_score,4))
feat_impdf['imp'] = model.feature_importances_


# In[ ]:


bkg = v_df[v_df['val_truth']==0]['val_pred_proba']
test1_proba = test_df['test1_pred_proba']
test2_proba = test_df['test2_pred_proba']

h_bkg = plt.hist(bkg,bins=np.arange(0,1.02,0.02), log=True, density=True)
h_test1 = plt.hist(test1_proba,bins=np.arange(0,1.02,0.02), log=True, density=True)
h_test2 = plt.hist(test2_proba,bins=np.arange(0,1.02,0.02), log=True, density=True)


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(h_test1[1][1:]-0.01, h_test1[0],'.', label='Test1',markersize='30', color='red')
plt.hist(x=bkg, bins=np.arange(0,1.02,0.02), histtype='step', label='Validation-BKG', rwidth='25',linewidth=3, log=True, density=True, 
         color='green')
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from BDT, 10k samples of each class, Test1', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=20)
plt.savefig('plot2.png')


plt.figure(figsize=(20,10))
plt.plot(h_test2[1][1:]-0.01, h_test2[0],'.', label='Test2',markersize='30', color='red')
plt.hist(x=bkg, bins=np.arange(0,1.02,0.02), histtype='step', label='Validation-BKG', rwidth='25',linewidth=3, log=True, density=True,
        color='green')
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from BDT, 10k samples of each class, Test2', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=20)
plt.savefig('plot3.png')


# In[ ]:


import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


# In[ ]:


def get_model():
    inputs = Input(shape=(7,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input=inputs, output=output)
    model.load_weights(MODEL_DIR+'model_name.h5')
    model.compile(optimizer='adam', metrics=['acc',AUC()], loss='binary_crossentropy')
    
    return model


# In[ ]:


get_model().summary()


# In[ ]:


class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1

        y_pred_val = self.model.predict(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[ ]:


nn_t_df = pd.DataFrame()
nn_v_df = pd.DataFrame()
nn_testdf = pd.DataFrame()
nn_t_df['train_truth'] = trn_Y
nn_t_df['train_predicted'] = 0
nn_t_df['train_pred_proba'] = 0
nn_v_df['val_truth'] = val_Y
nn_v_df['val_predicted'] = 0
nn_v_df['val_pred_proba'] = 0

nn_testdf['test1_pred_proba'] = 0
nn_testdf['test2_pred_proba'] = 0


nn_model = get_model()


# In[ ]:


nn_v_df['val_pred_proba'] = nn_model.predict(val_X)
val_pred_proba = nn_model.predict(val_X)
nn_testdf['test1_pred_proba'] = nn_model.predict(test1_X, batch_size=32, verbose=1).reshape(-1,)
nn_testdf['test2_pred_proba'] = nn_model.predict(test2_X, batch_size=32, verbose=1).reshape(-1,)
test3_pred_proba = nn_model.predict(test3_X, batch_size=32, verbose=1).reshape(-1,)
bkg3_pred_proba = nn_model.predict(bkg3_X, batch_size=32, verbose=1).reshape(-1,)
test4_pred_proba = nn_model.predict(test4_X, batch_size=32, verbose=1).reshape(-1,)


# In[ ]:


bkg = nn_v_df[nn_v_df['val_truth']==0]['val_pred_proba']
test1_proba = nn_testdf['test1_pred_proba']
test2_proba = nn_testdf['test2_pred_proba']


h_bkg = plt.hist(bkg,bins=np.arange(0,1.02,0.02), log=True, density=True)
h_test1 = plt.hist(test1_proba,bins=np.arange(0,1.02,0.02), log=True, density=True)
h_test2 = plt.hist(test2_proba,bins=np.arange(0,1.02,0.02), log=True, density=True)
h_test3 = plt.hist(test3_pred_proba,bins=np.arange(0,1.02,0.02), log=True, density=True)
h_bkg3 = plt.hist(bkg3_pred_proba,bins=np.arange(0,1.02,0.02), log=True, density=True)
h_test4 = plt.hist(test4_pred_proba,bins=np.arange(0,1.02,0.02), log=True, density=True)


# In[ ]:


plt.figure(figsize=(20,10))
plt.plot(h_test4[1][1:]-0.01, h_test4[0],'.', label='Test4',markersize='30', color='red')
plt.hist(x=bkg3_pred_proba, bins=np.arange(0,1.02,0.02), histtype='step', label='BKG3', rwidth='25',linewidth=3, log=True, density=True, 
         color='green')
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from NN, 10k samples of each class, Test4', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=20)
plt.savefig('nn_test4_10k.png')


plt.figure(figsize=(20,10))
plt.plot(h_test2[1][1:]-0.01, h_test2[0],'.', label='Test2',markersize='30', color='red')
plt.hist(x=bkg, bins=np.arange(0,1.02,0.02), histtype='step', label='Validation-BKG', rwidth='25',linewidth=3, log=True, density=True,
        color='green')
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from NN, 10k samples of each class, Test2', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=20)
plt.savefig('nn_test2_10k.png')

