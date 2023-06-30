#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from PIL import ImageOps    
import time
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torchsummary import summary
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import torchvision.models as models
from math import sqrt
from sklearn.metrics import *


# In[2]:


class SiameseNetworkDataset(Dataset):
    def __init__(self,pairs_df,img_dir = '/perfios/DATA/writer_verfication/test_cropped/',transform=None):
        self.pairs_df = pairs_df
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self,index):
        img1_name = self.pairs_df.iloc[index]['img1_name']
        img2_name = self.pairs_df.iloc[index]['img2_name']
        

        img1 = Image.open(self.img_dir+img1_name)
        img2 = Image.open(self.img_dir+img2_name)

        img2 = img2.convert("RGB")
        img1 = img1.convert("RGB")
        
        if self.transform is not None:
            img2 = self.transform(img2)
            img1 = self.transform(img1)

        
        return img1, img2, img1_name, img2_name

    def __len__(self):
        return len(self.pairs_df)


# In[3]:


def get_dataset_loader(df, transformation, batch_size, shuffle=True):
  siamese_dataset = SiameseNetworkDataset(pairs_df=df,transform=transformation)
  # Create a simple dataloader just for simple visualization
  dataloader = DataLoader(siamese_dataset,num_workers = 8,shuffle=shuffle,batch_size=batch_size, pin_memory=True)
  return siamese_dataset, dataloader


# In[4]:


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()

        self.features = torchvision.models.mobilenet_v3_small()    
        self.fc_in_features = self.features.classifier[0].in_features        
        self.features = torch.nn.Sequential(*(list(self.features.children())[:-2]))
        self.self_attention = nn.MultiheadAttention(self.fc_in_features, num_heads = 3)
        self.classifier = nn.Linear(self.fc_in_features, 1)


    def forward_once(self, x):
        x = self.features(x)
        b,f,h,w = x.shape
        x = x.view(b,w*h,f)   
        x,_ = self.self_attention(x,x,x)
        return x.squeeze()

    def forward(self, input1, input2):
        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        diff = torch.abs(output1 - output2)
        
        diff = torch.mean(diff, dim=1)

        diff = nn.ReLU()(diff)
        
        out = self.classifier(diff)
        
        return out


# In[5]:


test_df = pd.read_csv('test.csv') # replace this with Test csv
# test_df = test_df.sample(n=100)


# In[6]:


transformation = transforms.Compose([transforms.Resize((128,512)),transforms.ToTensor(),  transforms.Normalize(mean=0, std=1)])
test_siamese_dataset, test_dataloader = get_dataset_loader(test_df, transformation, batch_size=16, shuffle=False)


# In[7]:


example_batch = next(iter(test_dataloader))
plt.imshow(example_batch[0][8].permute(1,2,0))


# In[8]:


plt.imshow(example_batch[1][8].permute(1,2,0))


# In[9]:


net = SiameseNetwork().cuda()

net.load_state_dict(torch.load('hindi_SA_v2.pth'))
net.eval()


# In[10]:


preds_prob = []
all_ids = [] 
with torch.no_grad():
    for (x1, x2, x1_paths, x2_paths) in tqdm(test_dataloader):
        
        ids = [x1_path+"_"+x2_path for x1_path, x2_path in zip(x1_paths, x2_paths)]
        
        sim = net(x1.cuda(), x2.cuda())
        
        predicted_prob = torch.sigmoid(sim.squeeze()).cpu().numpy()      
        preds_prob.extend(predicted_prob)        
        all_ids.extend(ids)


# In[11]:


predicted_prob


# In[12]:


submission_df = pd.DataFrame(columns = ['id','proba'])
submission_df['id'] = pd.Series(all_ids)
submission_df['proba'] = pd.Series(preds_prob)


# In[13]:


submission_df


# In[14]:


submission_df.to_csv('submission.csv',index=False)


# In[15]:


submission_df.proba.mean()


# In[ ]:




