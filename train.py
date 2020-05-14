#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import os
import random
import seaborn as sns
from radam import *
from newlayers import *
import cv2
import matplotlib.pyplot as plt
import albumentations.augmentations.transforms as A
import albumentations.imgaug.transforms as IA
from albumentations import IAAAffine
from albumentations.core.composition import Compose
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import normalize
from tqdm.notebook import tqdm
import torchvision.models as models
import torch
from torch.utils.data import Dataset,DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import torch.optim 
from torch import nn

sns.set_style('dark')


# In[39]:


IMAGE_DIR = 'path/to/saved/images/'
MODEL_DIR = 'path/to/saved/models/'
EVAL_DIR = 'path/to/save/plots/'


# In[40]:


def make_df(folder):
    path_list = [os.path.join(IMAGE_DIR,folder,file) for file in os.listdir(IMAGE_DIR+folder)]
    labels = [int('pass' in name) for name in os.listdir(IMAGE_DIR+folder)]
    df = pd.DataFrame(list(zip(path_list, labels)), columns=['path','label'])
    
    return df


# In[41]:


def prepare_data(seed, n_samples=5000):
    dytoee_df = make_df('DYToEE_halved_Y')
    dytoee_df = dytoee_df.sample(n=n_samples, random_state=seed)
    qcd_df = make_df('QCD_halved_Y')
    qcd_df = qcd_df.sample(n=n_samples, random_state=seed)
    df = pd.concat([dytoee_df, qcd_df],0)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    traindf, valdf = train_test_split(df, test_size=config.TEST_SIZE, random_state=seed, stratify=df['label'])
    traindf.reset_index(drop=True, inplace=True)
    valdf.reset_index(drop=True, inplace=True)
    
    return traindf, valdf


# In[46]:


class Lepton(Dataset):
    def __init__(self, df, transform, mode='train'):
        self.path = df['path'].values
        self.label = df['label'].values
        self.transform = transform
        self.mode=mode
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        img_path = self.path[idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = self.label[idx]
        img=normalize(img.reshape(-1,1), norm='max', axis=0).reshape(47,47)
        
        if config.AUGMENTATIONS:
            if self.mode=='train':
                img = self.transform(image=img)['image']
        

        img = img[np.newaxis,:,:]
            
        return {'img': torch.tensor(img, dtype=torch.float),
               'label': torch.tensor(label, dtype=torch.float)
               }


# In[51]:


class PaperCNN(nn.Module):   #The best
    def __init__(self):
        super(PaperCNN, self).__init__()
        
        self.layer1 = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(1,64,8,padding=0, stride=1, bias=False)),
                                                ('batchnorm1', nn.BatchNorm2d(64)),
                                                ('relu1', nn.ReLU(inplace=True)),
                                                ('maxpool1', nn.MaxPool2d((2,2), padding=0, stride=2))])) 
        self.layer2 = nn.Sequential(OrderedDict([('conv2', nn.Conv2d(64,64,4,padding=0, stride=1, bias=False)),
                                                ('batchnorm2', nn.BatchNorm2d(64)),
                                                ('relu2', nn.ReLU(inplace=True)),
                                                ('maxpool2', nn.MaxPool2d((2,2), padding=0, stride=2))]))
        self.layer3 = nn.Sequential(OrderedDict([('conv3', nn.Conv2d(64,64,4,padding=0, stride=1, bias=False)),
                                                ('batchnorm3', nn.BatchNorm2d(64)),
                                                ('relu3', nn.ReLU(inplace=True)) ]))          
        
        
        self.avg = nn.AdaptiveAvgPool2d(output_size=2)
        self.max = nn.AdaptiveMaxPool2d((2,2))
        
        self.linear = nn.Linear(512,128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(128,1)
        
        
        # Kaiming method for initialization of weights
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        a = self.avg(x)
        m = self.max(x)
        x = torch.cat((a,m), 1)
        x=x.view(x.size(0),-1)
        x = self.linear(x)
        x=self.relu(x)
        x=self.dropout(x)
        out=self.linear2(x)
        out = out.squeeze(1)
        
        return out


# In[52]:


def train(loader, scheduler, threshold, pretrained=False):
    model.train()
    train_loss=0
    train_proba_pred, train_score, auc_score=[], [], []
    train_labels=[]
    
    optimizer.zero_grad()
    
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        img, label = batch['img'].to(config.DEVICE), batch['label'].to(config.DEVICE)
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        if not pretrained:
            if config.GRAD_ACCUM:
                if (i%config.GRAD_STEP)==0:
                    optimizer.step()
                    optimizer.zero_grad()        
            else:
                optimizer.step()
                optimizer.zero_grad()
        if pretrained:
            optimizer.zero_grad()
            
        if scheduler:
            scheduler.step()
        

        proba = torch.sigmoid(output.detach())          
        proba = proba.cpu().numpy()                             
        proba_threshold = (proba>threshold).astype('int')        
        score = metric(label.detach().cpu().numpy(), proba_threshold)    

        train_proba_pred.extend(proba)
        auc = roc_auc_score(label.detach().cpu().numpy(), proba)
        auc_score.append(auc)
        train_score.append(score)
        train_labels.extend(label.detach().cpu().numpy())
        train_loss+=loss.item()/len(loader)
        
    avg_score = np.mean(train_score)
    avg_auc= np.mean(auc_score)    
        
    return train_loss, train_proba_pred, avg_score, train_labels, avg_auc


# In[53]:


def validate(loader, scheduler, threshold):
    model.eval()
    val_loss=0
    val_proba_pred, all_scores, auc_score=[], [], []
    val_labels=[]
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            img, label = batch['img'].to(config.DEVICE), batch['label'].to(config.DEVICE)
            output = model(img)
            loss = criterion(output, label)
            val_loss+=loss.item()/len(loader)
            
            proba = torch.sigmoid(output.detach())                          
            proba = proba.cpu().numpy()                                        
            proba_threshold = (proba>threshold).astype('int')                    
            score = metric(label.detach().cpu().numpy(), proba_threshold)       

            val_proba_pred.extend(proba)
            auc = roc_auc_score(label.detach().cpu().numpy(), proba)
            auc_score.append(auc)
            all_scores.append(score)
            val_labels.extend(label.detach().cpu().numpy())
        
    if scheduler:
        scheduler.step(val_loss)
        
    avg_score=np.mean(all_scores)  
    avg_auc = np.mean(auc_score)
    
    return val_loss, val_proba_pred, avg_score, val_labels, avg_auc


# In[54]:


def run(num_epochs, seed, pretrained=False, model_name=None):    
    train_loss, val_loss = [], []
    train_score, val_score= [], []
    train_auc, val_auc = [], []
    
    best_loss = np.inf
    best_score=0
    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        
        print("Training Started .....")
        loss, tproba, tscore, tlabels, tauc = train(trainloader, None, config.THRESHOLD, pretrained)
        train_loss.append(loss)
        train_score.append(tscore)
        train_auc.append(tauc)
            
            
        print("Validation Started .....")    
        loss, vproba, vscore, vlabels, vauc = validate(valloader, scheduler, config.THRESHOLD)
        val_loss.append(loss)
        val_score.append(vscore)
        val_auc.append(vauc)

            
        print(f"Training accuracy: {tscore}, Validation Accuracy: {vscore}")
        print(f"Training ROCAUC: {tauc}, Validation ROCAUC: {vauc}")
        
        if (loss<best_loss):
            print(f"Validation loss decreased from {best_loss:.4f} to {loss:.4f}")
            print("Saving model...")
            if not pretrained:
                torch.save(model.state_dict(), MODEL_DIR + f'{model_name}_loss.pt')
                
            best_loss = loss

            t_auc, v_auc, t_acc, v_acc = tauc, vauc, tscore, vscore
        
            
    return {'train_loss': np.array(train_loss),
           'train_acc_array': np.array(train_score),
            'train_auc_array': np.array(train_auc),
            'train_proba_pred': np.array(tproba),
            'train_labels': np.array(tlabels),
            'train_auc': t_auc,
            'train_acc': t_acc,
           'val_loss': np.array(val_loss),
           'val_acc_array': np.array(val_score),
            'val_auc_array': np.array(val_auc),
            'val_proba_pred': np.array(vproba),
            'val_labels': np.array(vlabels),
            'val_auc': v_auc,
            'val_acc': v_acc}  


# In[55]:


def make_plots(rows=3, cmap='Greys_r'):
    fig,ax=plt.subplots(rows,3, figsize=(rows*5,15))
    for i in range(3):
        for j in range(rows):
            index = random.randint(0, len(traindf))       
            ax[j,i].imshow(traindataset[index]['img'].numpy().squeeze(0), cmap=cmap)
            ax[j,i].set_title(f"Label: {traindataset[index]['label'].numpy().astype('int')}")
#             ax[j,i].set_colorbar()
    plt.tight_layout()
    plt.show()


# In[56]:


# Make parameter changes over here

class Config:
    SEED=2020        
    TEST_SIZE=0.5               # Split Total Data by this factor to form Val and Train set
    AUGMENTATIONS=False         #Set True for Augmentations
    GRAD_ACCUM=False             # set True if OOM
    GRAD_STEP=4
    EPOCHS=15
    LR=1e-3
    BATCH_SIZE=64
    RLP_PATIENCE=5       #RLP is ReduceLROnPlateau
    RLP_FACTOR=0.5
    DEVICE='cuda:0'     # GPU
    
config = Config


# In[57]:


#Set seed for reproducible results

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(config.SEED)


# In[58]:


traindf, valdf = prepare_data(config.SEED, n_samples=5000)

augmentation = Compose([A.HorizontalFlip(p=0.5)])

traindataset = Lepton(traindf,augmentation, mode='train')
valdataset = Lepton(valdf,None, mode='validation')

# Iterator to make batches
trainloader = DataLoader(traindataset, batch_size=config.BATCH_SIZE, sampler=RandomSampler(traindataset))
valloader = DataLoader(valdataset, batch_size=config.BATCH_SIZE, sampler=SequentialSampler(valdataset))

# Initialise Model
model = PaperCNN()

model = model.to(config.DEVICE)

# Loss function
criterion = nn.BCEWithLogitsLoss() 

# Optimizer to optimize the loss functions
optimizer = torch.optim.AdamW(model.parameters(), lr = config.LR, weight_decay=0.0001)

# Scheduler to vary learning rate
scheduler = ReduceLROnPlateau(optimizer, factor=config.RLP_FACTOR, patience=config.RLP_PATIENCE, verbose=True)

metric = accuracy_score


# In[60]:


make_plots(cmap='Greys_r')


# In[61]:


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
ax1.bar(x= traindf['label'].value_counts().index, height=traindf['label'].value_counts().values, width=0.2)
ax1.set_title('Training labels distribution')
ax2.bar(x= valdf['label'].value_counts().index, height=valdf['label'].value_counts().values, width=0.2)
ax2.set_title('Validation labels distribution')


# In[62]:


MODEL_NAME='write your model name'
Title = 'write title for plots'
history = run(config.EPOCHS, config.SEED, pretrained=False, model_name=MODEL_NAME)


# In[63]:


print(history['train_acc_array'])
print(history['val_acc_array'])
print(history['train_auc_array'])
print(history['val_auc_array'])


# In[64]:


best_auc=history['val_auc']
best_acc=history['val_acc']
auc=history['val_auc_array'][-1]
acc=history['val_acc_array'][-1]


# In[65]:


# Needed for probability distribution

real_proba=history['train_proba_pred'][np.where(history['train_labels']==1)]
fake_proba=history['train_proba_pred'][np.where(history['train_labels']==0)]


val_real_proba=history['val_proba_pred'][np.where(history['val_labels']==1)]
val_fake_proba=history['val_proba_pred'][np.where(history['val_labels']==0)]

import matplotlib.pyplot as plt

val_dy_hist = plt.hist(val_real_proba,bins=np.arange(0,1.2,0.2))
val_qcd_hist = plt.hist(val_fake_proba,bins=np.arange(0,1.2,0.2))
plt.close()

yerr_dy = np.sqrt(val_dy_hist[0])
yerr_qcd = np.sqrt(val_qcd_hist[0])

y, binedges = np.histogram(real_proba, bins=np.arange(0,1.2,0.2))
y_fake, binedges = np.histogram(fake_proba, bins=np.arange(0,1.2,0.2))
bincenters=0.5*(binedges[1:]+binedges[:-1])


# In[67]:


# Make probability Distribution

plt.figure(figsize=(20,10))
plt.errorbar(x=val_dy_hist[1][1:]-0.1, y=val_dy_hist[0],yerr=yerr_dy,fmt='.', elinewidth=3,
             label='Val_Real',markersize='30',color='green')
plt.errorbar(x=val_qcd_hist[1][1:]-0.1, y=val_qcd_hist[0],yerr=yerr_qcd,fmt='.', elinewidth=3, 
             label='Val_Fake',markersize='30',color='red')
plt.bar(bincenters, y, yerr=np.sqrt(y), label='Train_Real',linewidth=3,edgecolor='green', width=0.2,
        fill=False, ecolor='green')
plt.bar(bincenters, y_fake, yerr=np.sqrt(y_fake),label='Train_Fake',linewidth=3,
        edgecolor='red', width=0.2, fill=False, ecolor='red')
plt.legend(loc='upper center', fontsize='x-large')
plt.xlabel('Output',fontsize=20)
plt.ylabel('Entries', fontsize=20)
plt.title(f'Histogram of Probability Distribution, Max Norm, {Title}, Acc:{acc:.4f}, AUC:{auc:.4f}',
          fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(EVAL_DIR+f'{MODEL_NAME}_probdist.png')


# In[68]:


# Loss, Accuracy and AUC vs Epochs Graphs

plt.figure(figsize=(45,15))

plt.subplot(1,3,1)
x_label = np.arange(1,config.EPOCHS+1, dtype='int')
plt.plot(x_label, history['train_loss'], linestyle='-.',linewidth=3, label='Train Loss')
plt.plot(x_label, history['val_loss'], linestyle='-.', linewidth=3, label='Val Loss')
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epoch Number', fontsize=30)
plt.ylabel('Loss', fontsize=30)
plt.title(f'Loss vs Epochs, Max Norm, {Title}', fontsize=20)

plt.subplot(1,3,2)
x_label = np.arange(1,config.EPOCHS+1, dtype='int')
plt.plot(x_label, history['train_acc_array'], linestyle='-.', linewidth=3, label='Train Acc')
plt.plot(x_label, history['val_acc_array'], linestyle='-.', linewidth=3, label='Val Acc')
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epoch Number', fontsize=30)
plt.ylabel('Accuracy', fontsize=30)
plt.title(f'Accuracy vs Epochs, Max Norm, {Title}, Acc:{acc:.4f}', fontsize=20)

plt.subplot(1,3,3)
x_label = np.arange(1,config.EPOCHS+1, dtype='int')
plt.plot(x_label, history['train_auc_array'], linestyle='-.', linewidth=3, label='Train AUC')
plt.plot(x_label, history['val_auc_array'], linestyle='-.', linewidth=3, label='Val AUC')
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epoch Number', fontsize=30)
plt.ylabel('ROCAUC', fontsize=30)
plt.title(f'ROC-AUC vs Epochs, Max Norm, {Title}, AUC:{auc:.4f}', fontsize=20)

plt.tight_layout()

plt.savefig(EVAL_DIR+f'{MODEL_NAME}.png')


# In[69]:


cnn_fpr, cnn_tpr, _ = roc_curve(history['val_labels'], history['val_proba_pred'])
ns_proba = [0]*2500
ns_fpr, ns_tpr, _ = roc_curve([0,1]*1250, ns_proba)


# In[73]:


# ROC-AUC Curve

plt.figure(figsize=(10,10))
plt.plot(cnn_fpr, cnn_tpr, linestyle='-.',linewidth=3, label='CNN Model')
plt.plot(ns_fpr, ns_tpr, linestyle='--',linewidth=3, label='No Skill')
plt.legend()
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title(f'ROC Curve, Max Norm, {Title}, AUC:{auc:.4f}, ACC:{acc:.4f}', fontsize=15)
plt.legend()
plt.savefig(EVAL_DIR+f'{MODEL_NAME}_roc.png')


# In[71]:


# performance={'fpr':[],'tpr':[],'t_loss':[],'v_loss':[],'t_acc':[],'v_acc':[], 't_auc':[], 'v_auc':[],
#              'acc':[], 'auc':[], 'best_auc':[], 'best_acc':[], 'title':[], 'model_name':[]}
performance = pd.read_pickle(EVAL_DIR+"name_of_file.pkl")

performance['fpr'].append(cnn_fpr)
performance['tpr'].append(cnn_tpr)
performance['t_loss'].append(history['train_loss'])
performance['v_loss'].append(history['val_loss'])
performance['t_acc'].append(history['train_acc_array'])
performance['v_acc'].append(history['val_acc_array'])
performance['t_auc'].append(history['train_auc_array'])
performance['v_auc'].append(history['val_auc_array'])
performance['best_acc'].append(best_acc)
performance['best_auc'].append(best_auc)
performance['acc'].append(acc)
performance['auc'].append(auc)
performance['title'].append(Title)
performance['model_name'].append(MODEL_NAME)

import pickle

f=open(EVAL_DIR+"name_of_file.pkl","wb")
pickle.dump(performance,f)
f.close

