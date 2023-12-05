import gc
import os
import random
import time
import warnings
warnings.simplefilter("ignore")

#import pdb
#import zipfile
#import pydicom
from albumentations import *
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import tifffile as tiff
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, sampler
from tqdm import tqdm_notebook as tqdm

%matplotlib inline

# Configuration
fold = 0
sz = 256
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 1

# Constants
EPOCHS = 15
SEED = 2020
TH = 0.5  # Threshold for positive predictions
LABELS_DIR = "/content/drive/MyDrive/MobileNet/Image224"
MASKS_DIR = '/content/drive/MyDrive/MobileNet/Mask224'
TRAIN_DIR = '/content/drive/MyDrive/MobileNet/Image224'
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

# Functions
def img2tensor(img, dtype=np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class ReadDataset(Dataset):
    def __init__(self, fold=fold, train=True, tfms=None):
        self.fnames = [fname for fname in os.listdir(TRAIN_DIR)]
        self.train = train
        self.tfms = tfms
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_DIR, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img2tensor((img / 255.0 - mean) / std), img2tensor(mask)

def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=p),
        VerticalFlip(p=p),
        RandomRotate90(p=p),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=p, border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(p=0.3),            
        ], p=0.3),
    ], p=p)

# Data loading
ds = ReadDataset(tfms=get_aug())
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Example of train images with masks
imgs, masks = next(iter(dl))

# Visualization
plt.figure(figsize=(16, 16))
for i, (img, mask) in enumerate(zip(imgs, masks)):
    img = ((img.permute(1, 2, 0) * std + mean) * 255.0).numpy().astype(np.uint8)
    plt.subplot(8, 8, i + 1)
    plt.imshow(img, vmin=0, vmax=255)
    plt.imshow(mask.squeeze().numpy(), alpha=0.2)
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)

plt.show()
del ds, dl, imgs, masks
def get_UnetPlusPlus():
    model =  smp.UnetPlusPlus(
                 encoder_name='efficientnet-b3',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=1)
    return model

#https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
cv_score = 0
for fold in range(nfolds):
    ds_t = HuBMAPDataset(fold=fold, train=True, tfms=get_aug())
    ds_v = HuBMAPDataset(fold=fold, train=False)
    dataloader_t = torch.utils.data.DataLoader(ds_t,batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)
    dataloader_v = torch.utils.data.DataLoader(ds_t,batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)
    model = get_UnetPlusPlus().to(DEVICE)
    
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-3}, 
        {'params': model.encoder.parameters(), 'lr': 1e-3},  
    ])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(dataloader_t))
    
    diceloss = DiceLoss()
    
    print(f"########FOLD: {fold}##############")
    
    for epoch in tqdm(range(EPOCHS)):
        ###Train
        model.train()
        train_loss = 0
    
        for data in dataloader_t:
            optimizer.zero_grad()
            img, mask = data
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
        
            outputs = model(img)
    
            loss = diceloss(outputs, mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        train_loss /= len(dataloader_t)
        
        print(f"FOLD: {fold}, EPOCH: {epoch + 1}, train_loss: {train_loss}")
        
        ###Validation
        model.eval()
        valid_loss = 0
        
        for data in dataloader_v:
            img, mask = data
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
        
            outputs = model(img)
    
            loss = diceloss(outputs, mask)
        
            valid_loss += loss.item()
        valid_loss /= len(dataloader_v)
        
        print(f"FOLD: {fold}, EPOCH: {epoch + 1}, valid_loss: {valid_loss}")
        
        
    ###Save model
    torch.save(model.state_dict(), f"FOLD{fold}_.pth")
    
    cv_score += valid_loss
    
cv_score = cv_score
