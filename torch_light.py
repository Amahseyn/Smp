!pip install -q segmentation-models-pytorch pytorch-lightning tabulate

import os
import random
import time
import warnings
warnings.simplefilter("ignore")

from albumentations import *
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pytorch_lightning as pl

# Configuration
fold = 0
nfolds = 5
reduce = 4
sz = 256
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 1

# Constants
EPOCHS = 100
SEED = 2020
TH = 0.5
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

# Your existing code...
class ReadDataset(Dataset):
    def __init__(self, images, fold=fold, train=True, tfms=None):
        self.fnames = images
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_DIR, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS_DIR, fname), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
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
            RandomGamma(p=0.3)
        ], p=0.3),
    ], p=p)

def get_UnetPlusPlus():
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1)
    return model
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
# PyTorch Lightning Module
# PyTorch Lightning Module
class SegmentationModel(pl.LightningModule):
    def __init__(self, model, optimizer_params, scheduler_params, diceloss):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.diceloss = diceloss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        outputs = self(img)
        loss = self.diceloss(outputs, mask)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        outputs = self(img)
        loss = self.diceloss(outputs, mask)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_params)
        scheduler = OneCycleLR(optimizer=optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]


# Data loading
train_image, val_image = train_test_split(os.listdir(TRAIN_DIR), test_size=0.2, random_state=SEED)
val_image, test_image = train_test_split(val_image, test_size=0.5, random_state=SEED)

# Training loop with PyTorch Lightning
ds_t = ReadDataset(train_image, fold=fold, train=True, tfms=get_aug())
ds_v = ReadDataset(val_image, fold=fold, train=False)
ds_test = ReadDataset(test_image, fold=fold, train=False)

# Model and Lightning Module Initialization
model = get_UnetPlusPlus()
optimizer_params = {'lr': 1e-4}
scheduler_params = {'pct_start': 0.1, 'div_factor': 1e3, 'max_lr': 1e-2, 'total_steps': len(ds_t) * EPOCHS, 'anneal_strategy': 'linear', 'cycle_momentum': False}
lightning_model = SegmentationModel(model, optimizer_params, scheduler_params, DiceLoss())

# PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=EPOCHS, accelerator='auto' if torch.cuda.is_available() else None)

# Training
trainer.fit(lightning_model, DataLoader(ds_t, batch_size=1, shuffle=True, num_workers=NUM_WORKERS),
            DataLoader(ds_v, batch_size=1, shuffle=False, num_workers=NUM_WORKERS))

# Testing
trainer.test(lightning_model, DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS))

# Visualization
visualization_dataloader = DataLoader(ds_v, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

with torch.no_grad():
    for i, data in enumerate(visualization_dataloader):
        img, mask = data
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        outputs = lightning_model(img)

        img_np = ((img.squeeze().permute(1, 2, 0).cpu().numpy() * std + mean) * 255.0).astype(np.uint8)
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = (outputs.squeeze().cpu().numpy() > TH).astype(np.uint8)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_np, cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred_np, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

        plt.show()
