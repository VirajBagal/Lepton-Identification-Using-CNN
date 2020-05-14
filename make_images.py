#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
OPTION=1


# In[20]:


DATA_DIR = "/path/to/text/file/"


# In[21]:


dytoee=[]

for file in os.listdir(DATA_DIR):
    df=pd.read_csv(os.path.join(DATA_DIR,file), sep=" ")
    df.columns = ["pt","eta","phi","energy","hcal1_et","hcal2_et","hcal1_etbc","hcal2_etbc","pfiso",
                      "chargediso","neutraliso","photoniso",
                     "tksumpt","ecal_rechit","calo_eta","calo_phi","calo_em","calo_had","calo_outer",
                     "calo_emet","calo_hadet","calo_outeret"]
    dytoee.append(df)
    break
    
dytoee=pd.concat(dytoee)


# In[23]:


count=0
for i,df in enumerate(dytoee.groupby(['pt','eta','phi'])):
    count+=1
    
print("Number of pass dytoee electrons: ", count)


# In[26]:


qcd=[]

for file in os.listdir(DATA_DIR):
    df=pd.read_csv(os.path.join(DATA_DIR,file), sep=" ")
    df.columns = ["pt","eta","phi","energy","hcal1_et","hcal2_et","hcal1_etbc","hcal2_etbc","pfiso",
                      "chargediso","neutraliso","photoniso",
                     "tksumpt","ecal_rechit","calo_eta","calo_phi","calo_em","calo_had","calo_outer",
                     "calo_emet","calo_hadet","calo_outeret"]
    qcd.append(df)


# In[27]:


qcd=pd.concat(qcd)

count=0
for i,df in enumerate(qcd.groupby(['pt','eta','phi'])):
    count+=1
    
print("Number of pass qcd electrons: ", count)


# In[291]:


def make_rgb(spacing=10):
    array = np.ones((460,256,3))
    for i in range(460):
        for j in range(256):
            if i//spacing==0: array[i,j,:] = [j,0,0]
            if i//spacing==1: array[i,j,:] = [0,j,0]
            if i//spacing==2: array[i,j,:] = [0,0,j] 
            if i//spacing==3: array[i,j,:] = [j,j,0]
            if i//spacing==4: array[i,j,:] = [j,0,j]
            if i//spacing==5: array[i,j,:] = [0,j,j]
            if i//spacing==6: array[i,j,:] = [j,j,j]
            if i//spacing==7: array[i,j,:] = [j,75,75]
            if i//spacing==8: array[i,j,:] = [75,j,75]
            if i//spacing==9: array[i,j,:] = [75,75,j] 
            if i//spacing==10: array[i,j,:] = [j,j,75]
            if i//spacing==11: array[i,j,:] = [j,75,j]
            if i//spacing==12: array[i,j,:] = [75,j,j]
            if i//spacing==13: array[i,j,:] = [j,128,128]
            if i//spacing==14: array[i,j,:] = [128,j,128]
            if i//spacing==15: array[i,j,:] = [128,128,j] 
            if i//spacing==16: array[i,j,:] = [j,j,128]
            if i//spacing==17: array[i,j,:] = [j,128,j]
            if i//spacing==18: array[i,j,:] = [128,j,j]
            if i//spacing==19: array[i,j,:] = [j,255,255]
            if i//spacing==20: array[i,j,:] = [255,j,255]
            if i//spacing==21: array[i,j,:] = [255,255,j] 
            if i//spacing==22: array[i,j,:] = [j,j,255]
            if i//spacing==23: array[i,j,:] = [j,255,j]
            if i//spacing==24: array[i,j,:] = [255,j,j]
            if i//spacing==25: array[i,j,:] = [75,128,0]
            if i//spacing==26: array[i,j:] = [75,0,128]
            if i//spacing==27: array[i,j,:] = [0,75,128]
            if i//spacing==28: array[i,j,:] = [128,75,0]
            if i//spacing==29: array[i,j:] = [128,0,75]
            if i//spacing==30: array[i,j,:] = [0,128,75]
            if i//spacing==31: array[i,j,:] = [75,255,0]
            if i//spacing==32: array[i,j:] = [75,0,255]
            if i//spacing==33: array[i,j,:] = [0,75,255]
            if i//spacing==34: array[i,j,:] = [225,75,0]
            if i//spacing==35: array[i,j:] = [225,0,75]
            if i//spacing==36: array[i,j,:] = [0,225,75]
            if i//spacing==37: array[i,j,:] = [128,255,0]
            if i//spacing==38: array[i,j:] = [128,0,255]
            if i//spacing==39: array[i,j,:] = [0,128,255]
            if i//spacing==40: array[i,j,:] = [255,128,0]
            if i//spacing==41: array[i,j:] = [255,0,128]
            if i//spacing==42: array[i,j,:] = [0,255,128]
            if i//spacing==43: array[i,j,:] = [75,128,255]
            if i//spacing==44: array[i,j,:] = [128,255,75]
            if i//spacing==45: array[i,j,:] = [255,75,128]
                
    return array.astype('uint8')


# In[292]:


yaxis_labels = ['(j,0,0)', '(0,j,0)', '(0,0,j)', '(j,j,0)', '(j,0,j)', '(0,j,j)',
               '(j,j,j)','(j,75,75)','(75,j,75)','(75,75,j)','(j,j,75)','(j,75,j)','(75,j,j)', 
                '(j,128,128)','(128,j,128)','(128,128,j)','(j,j,128)','(j,128,j)','(128,j,j)',
                '(j,225,225)','(225,j,225)','(225,225,j)','(j,j,225)','(j,225,j)','(225,j,j)',
                '(75,128,0)','(75,0,128)','(0,75,128)','(128,75,0)','(128,0,75)', '(0,128,75)',
               '(75,255,0)','(75,0,255)','(0,75,255)','(225,75,0)','(225,0,75)','(0,255,75)',
               '(128,255,0)','(128,0,255)','(0,128,255)','(255,128,0)','(255,0,128)','(0,255,128)',
               '(75,128,255)','(128,255,75)','(255,75,128)']


# In[316]:


test = make_rgb()
plt.figure(figsize=(15,25))
plt.imshow(test)
locs, label = plt.yticks()
locs = np.arange(0,470,10)
labels = yaxis_labels
plt.yticks(locs, labels)
# plt.axis('off')
# plt.yticks(labels=yticks)
plt.savefig(SAVE_DIR+'test_rgb.png')


# In[35]:


def make_array(df, mode='ecal' norm='max'):
    pt = df['pt']
    eta = df['eta'].values
    phi = df['phi'].values
    calo_eta = df['calo_eta'].values
    calo_phi = df['calo_phi'].values
    deta = calo_eta-eta
    dphi = calo_phi-phi
    
    if mode=='ecal':
        calo_em = df['calo_em'].values       
    if mode=='hcal':
        calo_had = df['calo_had'].values
    else:
        calo_total = df['calo_em'].values+df['calo_had'].values+df['calo_outer'].values

    if norm:
        calo_total=normalize(calo_total.reshape(-1,1), norm=norm,axis=0).reshape(-1,)

            
    pixel_x = 22+(deta/0.0174).astype('int')
    pixel_y = 22+(dphi/0.0174).astype('int')
    
    coordinates=list(zip(pixel_x,pixel_y))
    filtered=[(x,y) for (x,y) in coordinates if ((x<47)&(y<23))&((x>=0)&(y>=0))]
    indices = [i for i,(x,y) in enumerate(coordinates) if (x,y) in filtered]
    newpixel_x, newpixel_y = zip(*filtered)
    calo_total=calo_total[indices]
        
#     array = np.zeros((47,47,3))      #3 channels
    
    array=np.zeros((47,47))
    
    if mode=='ecal': 
        array[newpixel_y, newpixel_x] = calo_em
    elif mode=='hcal': 
        array[newpixel_y, newpixel_x] = calo_had
    elif mode=='total':
        array[newpixel_y, newpixel_x] = calo_total
                
        
#     array[:,:,1]=0   #2nd channel
#     array[:,:,2]=0   #3rd channel

    array = array.astype('uint8')
    
    return array


# In[38]:


SAVE_DIR = 'path/to/save/directory'


# In[39]:


for i, df in enumerate(tqdm(dytoee.groupby(['pt','eta','phi']))):
    img=make_array(df[1], mode='total', norm=None)
    cv2.imwrite(SAVE_DIR+f'pass1_{i}.png', img)
    if i==5000:
        break

