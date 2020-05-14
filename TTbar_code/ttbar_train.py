#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


FILES_DIR = 'path/to/train/files'
seed=42     # For reproducible results
total_samples = 1000000
num_samples = 200000
folds=[0]


# In[3]:


column_names=['pt1','pt2','invmass','dphi','met','dphi1_met','dphi2_met']
ttbar = pd.concat([pd.read_csv(FILES_DIR+file,sep=' ', index_col=None, usecols=[1,2,3,4,5,6,7], names=column_names) for file in os.listdir(FILES_DIR) if 'ttbar' in file]).reset_index(drop=True)
sig = pd.concat([pd.read_csv(FILES_DIR+file,sep=' ', index_col=None,usecols=[1,2,3,4,5,6,7], names=column_names) for file in os.listdir(FILES_DIR) if 'sig' in file]).reset_index(drop=True)


# In[4]:


ttbar['dphi']=abs(ttbar['dphi'])


# In[5]:


ttbar.head()


# In[6]:


ttbar['fold']=-1
sig['fold']=-1
n = int(total_samples/num_samples)

for i in range(n):
    ttbar.loc[num_samples*i:num_samples*(i+1),'fold']=i
    sig.loc[num_samples*i:num_samples*(i+1),'fold']=i
    
ttbar['label']=0
sig['label']=1

data=pd.concat([ttbar,sig]).sample(frac=1, random_state=seed).reset_index(drop=True)


# In[7]:


def dist_plot(feat):
#     plt.figure(figsize=(8,8))
    ax=sns.distplot(data[data['label']==1][feat], label='Signal', norm_hist=True)
    ax=sns.distplot(data[data['label']==0][feat], label='Background', norm_hist=True)
    plt.legend()
    plt.xlabel(feat, fontsize=20)
    plt.ylabel('Normalized', fontsize=20)
    plt.title(f'Distribution of {feat}', fontsize=20)


# In[8]:


plt.figure(figsize=(16, 24))

for i in np.arange(0,5,2):
    plt.subplot(4,2,i+1)
    dist_plot(column_names[i])
    plt.subplot(4,2,i+2)
    dist_plot(column_names[i+1])
    
plt.subplot(4,2,7)
dist_plot(column_names[-1])
plt.savefig(f'{column_names[-1]}.png')

plt.tight_layout()


# In[9]:


traindf = data[data['fold'].isin(folds)].reset_index(drop=True)
trn_X = traindf.drop(['label','fold'],1)

# preprocessing done. No effect on BDT. Must for NN. 
mean, std = trn_X.mean(axis=0), trn_X.std(axis=0)
trn_X = (trn_X - mean)/std
trn_X = trn_X.values
trn_Y = traindf['label'].values
valdf =  data[~data['fold'].isin(folds)].reset_index(drop=True)
val_X = valdf.drop(['label','fold'],1)

val_X = (val_X - mean)/std   # Validation data statistics not used. Training statistics itself used.
val_X = val_X.values
val_Y = valdf['label'].values


# In[10]:


print('Train shape: ', len(trn_X))
print('Validation shape: ', len(val_X))


# In[11]:


t_df = pd.DataFrame()
v_df = pd.DataFrame()
t_df['train_truth'] = trn_Y
t_df['train_predicted'] = 0
t_df['train_pred_proba'] = 0
v_df['val_truth'] = val_Y
v_df['val_predicted'] = 0
v_df['val_pred_proba'] = 0
feat_impdf = pd.DataFrame()
feat_impdf['features'] = column_names

model = XGBClassifier(random_state=seed, n_estimators=500)
model.fit(trn_X, trn_Y, eval_set=[(val_X, val_Y)], eval_metric=['auc'], verbose=10, early_stopping_rounds=30)
val_pred_proba = model.predict_proba(val_X)[:,1] # predicts probability of the event being Signal
train_pred_proba = model.predict_proba(trn_X)[:,1]
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


# In[12]:


plt.figure(figsize=(10,10))
sns.barplot(x = feat_impdf['imp'], y=feat_impdf['features'])
plt.title('Feature Importance from BDT', fontsize=20)
plt.xlabel('Importance Value', fontsize=20)
plt.ylabel('Features', fontsize=20)
plt.savefig('plot1.png')


# In[13]:


xgb_fpr, xgb_tpr, _ = roc_curve(v_df['val_truth'].values, v_df['val_pred_proba'])
ns_fpr, ns_tpr, _ = roc_curve(v_df['val_truth'].values, [0]*len(v_df))

plt.figure(figsize=(10,10))
plt.plot(xgb_fpr, xgb_tpr, linestyle='-.', label='BDT')
plt.plot(ns_fpr, ns_tpr, linestyle='-.', label='No Skill')
plt.legend()
plt.title('BDT ROC Curve, 200k samples of each class', fontsize=20)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=20)
plt.savefig('plot2.png')


# In[14]:


import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import AUC
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


# In[15]:


def get_model():
    inputs = Input(shape=(7,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input=inputs, output=output)
    model.compile(optimizer='adam', metrics=['acc',AUC()], loss='binary_crossentropy')
    
    return model


# In[16]:


get_model().summary()


# In[17]:


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


# In[18]:


nn_t_df = pd.DataFrame()
nn_v_df = pd.DataFrame()
nn_t_df['train_truth'] = trn_Y
nn_t_df['train_predicted'] = 0
nn_t_df['train_pred_proba'] = 0
nn_v_df['val_truth'] = val_Y
nn_v_df['val_predicted'] = 0
nn_v_df['val_pred_proba'] = 0

cb = [roc_auc_callback(training_data=(trn_X, trn_Y), validation_data=(val_X, val_Y)),
      EarlyStopping(patience=3, verbose=1),
     ModelCheckpoint(filepath='best_model.h5')]

nn_model = get_model()
history=nn_model.fit(trn_X, trn_Y, epochs=15, batch_size=256, validation_data=(val_X,val_Y), callbacks=cb)


# In[19]:


history.history.keys()


# In[20]:


plt.figure(figsize=(10,30))


nepochs = np.arange(1,16)
plt.subplot(3,1,1)
plt.plot(nepochs, history.history['val_loss'], label='Validation')
plt.plot(nepochs, history.history['loss'], label='Train')
plt.legend()
plt.title('Loss vs Epochs', fontsize=20)
plt.xlabel('NEpochs', fontsize=20)
plt.ylabel('Loss')


nepochs = np.arange(1,16)
plt.subplot(3,1,2)
plt.plot(nepochs, history.history['val_acc'], label='Validation')
plt.plot(nepochs, history.history['acc'], label='Train')
plt.legend()
plt.title('Accuracy vs Epochs', fontsize=20)
plt.xlabel('NEpochs', fontsize=20)
plt.ylabel('Accuracy')


nepochs = np.arange(1,16)
plt.subplot(3,1,3)
plt.plot(nepochs, history.history['roc_auc_val'], label='Validation')
plt.plot(nepochs, history.history['roc_auc'], label='Train')
plt.legend()
plt.title('ROC-AUC vs Epochs', fontsize=20)
plt.xlabel('NEpochs', fontsize=20)
plt.ylabel('ROC-AUC')
plt.savefig('plot3.png')


# In[21]:


nn_t_df['train_pred_proba'] = nn_model.predict(trn_X)
nn_v_df['val_pred_proba'] = nn_model.predict(val_X)

nn_t_df['train_predicted'] = (nn_t_df['train_pred_proba'] > 0.5).values.astype('int')
nn_v_df['val_predicted'] = (nn_v_df['val_pred_proba'] > 0.5).values.astype('int')

trn_score = roc_auc_score(trn_Y, nn_t_df['train_pred_proba'].values)
val_score = roc_auc_score(val_Y, nn_v_df['val_pred_proba'].values)

print('Train ROC AUC: ', np.round(trn_score, 4))
print('Validation ROC AUC: ', np.round(val_score, 4))

trn_score = accuracy_score(trn_Y, nn_t_df['train_predicted'].values)
val_score = accuracy_score(val_Y, nn_v_df['val_predicted'].values)

print('Train Accuracy: ', np.round(trn_score, 4))
print('Validation Accuracy: ', np.round(val_score, 4))


# In[22]:


nn_fpr, nn_tpr, _ = roc_curve(nn_v_df['val_truth'].values, nn_v_df['val_pred_proba'])
ns_fpr, ns_tpr, _ = roc_curve(nn_v_df['val_truth'].values, [0]*len(nn_v_df))

plt.figure(figsize=(10,10))
plt.plot(nn_fpr, nn_tpr, linestyle='-.', label='Neural Network')
plt.plot(ns_fpr, ns_tpr, linestyle='-.', label='No Skill')
plt.legend()
plt.title('NN ROC Curve, 200k samples of each class', fontsize=20)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=20)

plt.savefig('plot4.png')


# In[23]:


plt.figure(figsize=(10,10))
plt.plot(nn_fpr, nn_tpr, linestyle='-.', label='Neural Network')
plt.plot(xgb_fpr, xgb_tpr, linestyle='-.', label='BDT')
plt.plot(ns_fpr, ns_tpr, linestyle='-.', label='No Skill')
plt.legend()
plt.title('ROC Curve, 200k samples of each class', fontsize=20)
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=20)

plt.savefig('plot5.png')


# In[24]:


fold_dict = {}
for i in valdf['fold'].unique():
    fold_dict[i]=valdf[valdf['fold']==i].index


# In[25]:


fold_dict.keys()


# In[26]:


xgb_bkg_dict={}
xgb_sig_dict={}

xgb_bkg_dict[0]=t_df[t_df['train_truth']==0]['train_pred_proba']
xgb_sig_dict[0]=t_df[t_df['train_truth']==1]['train_pred_proba']
for i in valdf['fold'].unique():
    df=v_df.iloc[fold_dict[i],:]
    xgb_bkg_dict[i]=df[df['val_truth']==0]['val_pred_proba']
    xgb_sig_dict[i]=df[df['val_truth']==1]['val_pred_proba']


# In[27]:


nn_bkg_dict={}
nn_sig_dict={}

nn_bkg_dict[0]=nn_t_df[nn_t_df['train_truth']==0]['train_pred_proba']
nn_sig_dict[0]=nn_t_df[nn_t_df['train_truth']==1]['train_pred_proba']
for i in valdf['fold'].unique():
    df=nn_v_df.iloc[fold_dict[i],:]
    nn_bkg_dict[i]=df[df['val_truth']==0]['val_pred_proba']
    nn_sig_dict[i]=df[df['val_truth']==1]['val_pred_proba']


# In[28]:


xgb_bkg1 = plt.hist(xgb_bkg_dict[1],bins=np.arange(0,1.02,0.02), log=True)
xgb_bkg2 = plt.hist(xgb_bkg_dict[2],bins=np.arange(0,1.02,0.02), log=True)
xgb_bkg3 = plt.hist(xgb_bkg_dict[3],bins=np.arange(0,1.02,0.02), log=True)
xgb_bkg4 = plt.hist(xgb_bkg_dict[4],bins=np.arange(0,1.02,0.02), log=True)

xgb_sig1 = plt.hist(xgb_sig_dict[1],bins=np.arange(0,1.02,0.02), log=True)
xgb_sig2 = plt.hist(xgb_sig_dict[2],bins=np.arange(0,1.02,0.02), log=True)
xgb_sig3 = plt.hist(xgb_sig_dict[3],bins=np.arange(0,1.02,0.02), log=True)
xgb_sig4 = plt.hist(xgb_sig_dict[4],bins=np.arange(0,1.02,0.02), log=True)


# In[29]:


get_ipython().run_line_magic('pinfo', 'plt.hist')


# In[30]:


plt.figure(figsize=(20,10))
plt.plot(xgb_bkg1[1][1:]-0.01, xgb_bkg1[0],'.', label='Valset_1',markersize='30')
plt.plot(xgb_bkg2[1][1:]-0.01, xgb_bkg2[0],'.', label='Valset_2',markersize='30')
plt.plot(xgb_bkg3[1][1:]-0.01, xgb_bkg3[0],'.', label='Valset_3',markersize='30')
plt.plot(xgb_bkg4[1][1:]-0.01, xgb_bkg4[0],'.', label='Valset_4',markersize='30')
plt.hist(x=(xgb_bkg_dict[0]), bins=np.arange(0,1.02,0.02), histtype='step', label='Train Set', rwidth='25',linewidth=3,color='green',
        log=True)
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from BDT, BKG', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('plot6.png')


# In[31]:


plt.figure(figsize=(20,10))
plt.plot(xgb_sig1[1][1:]-0.01, xgb_sig1[0],'.', label='Valset_1',markersize='30')
plt.plot(xgb_sig2[1][1:]-0.01, xgb_sig2[0],'.', label='Valset_2',markersize='30')
plt.plot(xgb_sig3[1][1:]-0.01, xgb_sig3[0],'.', label='Valset_3',markersize='30')
plt.plot(xgb_sig4[1][1:]-0.01, xgb_sig4[0],'.', label='Valset_4',markersize='30')
plt.hist(x=xgb_sig_dict[0], bins=np.arange(0,1.02,0.02), histtype='step', label='Train Set', rwidth='25',linewidth=3,color='green',
        log=True)
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from BDT, Signal', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('plot7.png')


# In[32]:


nn_bkg1 = plt.hist(nn_bkg_dict[1],bins=np.arange(0,1.02,0.02), log=True)
nn_bkg2 = plt.hist(nn_bkg_dict[2],bins=np.arange(0,1.02,0.02), log=True)
nn_bkg3 = plt.hist(nn_bkg_dict[3],bins=np.arange(0,1.02,0.02), log=True)
nn_bkg4 = plt.hist(nn_bkg_dict[4],bins=np.arange(0,1.02,0.02), log=True)

nn_sig1 = plt.hist(nn_sig_dict[1],bins=np.arange(0,1.02,0.02), log=True)
nn_sig2 = plt.hist(nn_sig_dict[2],bins=np.arange(0,1.02,0.02), log=True)
nn_sig3 = plt.hist(nn_sig_dict[3],bins=np.arange(0,1.02,0.02), log=True)
nn_sig4 = plt.hist(nn_sig_dict[4],bins=np.arange(0,1.02,0.02), log=True)


# In[33]:


plt.figure(figsize=(20,10))
plt.plot(nn_bkg1[1][1:]-0.01, nn_bkg1[0],'.', label='Valset_1',markersize='30')
plt.plot(nn_bkg2[1][1:]-0.01, nn_bkg2[0],'.', label='Valset_2',markersize='30')
plt.plot(nn_bkg3[1][1:]-0.01, nn_bkg3[0],'.', label='Valset_3',markersize='30')
plt.plot(nn_bkg4[1][1:]-0.01, nn_bkg4[0],'.', label='Valset_4',markersize='30')
plt.hist(x=nn_bkg_dict[0], bins=np.arange(0,1.02,0.02), histtype='step', label='Train Set', rwidth='25',linewidth=3,color='green',
        log=True)
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from NN, BKG', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('plot8.png')


# In[34]:


plt.figure(figsize=(20,10))
plt.plot(nn_sig1[1][1:]-0.01, nn_sig1[0],'.', label='Valset_1',markersize='30')
plt.plot(nn_sig2[1][1:]-0.01, nn_sig2[0],'.', label='Valset_2',markersize='30')
plt.plot(nn_sig3[1][1:]-0.01, nn_sig3[0],'.', label='Valset_3',markersize='30')
plt.plot(nn_sig4[1][1:]-0.01, nn_sig4[0],'.', label='Valset_4',markersize='30')
plt.hist(x=nn_sig_dict[0], bins=np.arange(0,1.02,0.02), histtype='step', label='Train Set', rwidth='25',linewidth=3,color='green',
         log=True)
plt.legend(loc='upper center')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution from NN, Signal', fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('plot9.png')

