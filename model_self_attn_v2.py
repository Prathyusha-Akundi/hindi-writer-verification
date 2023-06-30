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

class SiameseNetworkDataset(Dataset):
    def __init__(self,pairs_df,img_dir = '/perfios/DATA/writer_verfication/dataset/Cropped_Training_Data/',transform=None):
        self.pairs_df = pairs_df
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self,index):
        img1_path = self.img_dir+self.pairs_df.iloc[index]['img1_name']
        img2_path = self.img_dir+self.pairs_df.iloc[index]['img2_name']
        pair = self.pairs_df.iloc[index]['label']

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        img2 = img2.convert("RGB")
        img1 = img1.convert("RGB")
        
        if self.transform is not None:
            img2 = self.transform(img2)
            img1 = self.transform(img1)

        
        return img1, img2, pair

    def __len__(self):
        return len(self.pairs_df)
    
    
def get_dataset_loader(df, transformation, batch_size, shuffle=True):
  siamese_dataset = SiameseNetworkDataset(pairs_df=df,transform=transformation)
  # Create a simple dataloader just for simple visualization
  dataloader = DataLoader(siamese_dataset,num_workers = 8,shuffle=shuffle,batch_size=batch_size, pin_memory=True)
  return siamese_dataset, dataloader


def start_training(net, criterion, optimizer, epochs, save_path, patience, train_dataloader, validation_dataloader):

  def train_step():
    train_loss = 0.0
    for  data in tqdm(train_dataloader):
        img0, img1 , label = data
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        optimizer.zero_grad()
        
        output = net(img0,img1)        
        loss = criterion(output.squeeze(),label.float())
#         print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss/len(train_dataloader)
    #print('in')
    return train_loss
  
  def validation_step():
    val_loss = 0.0
    
    with torch.no_grad():
        for data in tqdm(validation_dataloader):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            output = net(img0,img1)        
            loss = criterion(output.squeeze(),label.float())
            val_loss += loss.item()
        
    val_loss = val_loss/len(validation_dataloader)
    return val_loss

  train_loss_history = [] 
  val_loss_history = [] 
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
  best_loss = float('inf')
  early_stopping_counter = 0

  for epoch in tqdm(range(epochs)):
    net.train()
    train_loss = train_step()
    net.eval()
    val_loss = validation_step()
    scheduler.step(val_loss)
    print(f"Epoch {epoch}\t Train Loss: {train_loss}\t Val Loss: {val_loss}")
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
#     torch.save(net.module.state_dict(), save_path)
      
    if val_loss < best_loss:
        print(f'Loss decreased from {best_loss} to {val_loss}. Saving the model at {save_path}')
        best_loss = val_loss
        
        torch.save(net.module.state_dict(), save_path)
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:            
            print("Early stopping")
            break
  return net, train_loss_history, val_loss_history


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()

        self.features = torchvision.models.mobilenet_v3_small()    
        self.fc_in_features = self.features.classifier[0].in_features        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.features = torch.nn.Sequential(*(list(self.features.children())[:-2]))
        self.self_attention = nn.MultiheadAttention(self.fc_in_features, num_heads = 3)#SelfAttention(self.fc_in_features)
        self.classifier = nn.Linear(self.fc_in_features, 1)


    def forward_once(self, x):
        x = self.features(x)

        b,f,h,w = x.shape
        x = x.view(b,w*h,f)   
#         x = x.permute(1,0,2)
        x,_ = self.self_attention(x,x,x)
        return x.squeeze()

    def forward(self, input1, input2):
        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
    
        
        if(torch.isnan(output1).any()):
            print('Output1 is Nan')
            
        if(torch.isnan(output2).any()):
            print('Output2 is Nan')


        diff = torch.abs(output1 - output2)
        
        diff = torch.mean(diff, dim=1)

        diff = nn.ReLU()(diff)
        
        out = self.classifier(diff)
        if(torch.isnan(out).any()):
            print('out is Nan')
        
        return out
    
    
pairs_df = pd.read_csv('/perfios/DATA/writer_verfication/train_cropped.csv')
# pairs_df = pairs_df.sample(n=10000, random_state=2023)
train_df, val_df = train_test_split(pairs_df, test_size=0.1, stratify=pairs_df['label'].values, random_state=2023, shuffle=True)

transformation = transforms.Compose([transforms.Resize((128,512)),transforms.ToTensor(),  transforms.Normalize(mean=0, std=1)])

train_df.to_csv('train_df_final.csv',index=False)
val_df.to_csv('val_df_final.csv',index=False)

train_siamese_dataset, train_dataloader = get_dataset_loader(train_df, transformation, batch_size=16)
val_siamese_dataset, val_dataloader = get_dataset_loader(val_df, transformation, batch_size=16, shuffle=True)

net = SiameseNetwork().cuda()
net = torch.nn.DataParallel(net)
criterion = nn.BCEWithLogitsLoss()  # since we are doing binary classification
optimizer = optim.Adam(net.parameters(), lr=0.0003)

net, train_loss_history, val_loss_history = start_training(net, criterion, optimizer, epochs=20, save_path='hindi_SA_v2.pth', patience=5, train_dataloader=train_dataloader, validation_dataloader=val_dataloader)

net = SiameseNetwork().cuda()

net.load_state_dict(torch.load('hindi_SA_v2.pth'))
net.eval()


preds = []
gt = []
with torch.no_grad():
    for (x1, x2, y) in tqdm(val_dataloader):
        sim = net(x1.cuda(), x2.cuda())
        if(torch.isnan(sim).any()):
            break
        predicted = torch.sigmoid(sim.squeeze()).data > 0.4
        preds.extend(predicted)
        gt.extend(y)
        
gt_ = np.array([int(i.numpy()) for i in gt])
preds_ = np.array([int(i.cpu().numpy()) for i in preds])

acc = np.sum(gt_==preds_)/len(gt_)
f1 = f1_score(gt_, preds_)
roc_auc = roc_auc_score(gt_, preds_)

print(f'Metrics on val set: Accuracy {acc}, F1-score {f1}, ROC AUC {roc_auc}')
