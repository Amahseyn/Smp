!pip install --root-user-action=ignore -q segmentation-models-pytorch pytorch-lightning tabulate
!pip install -U scikit-image
!pip install -U git+https://github.com/albumentations-team/albumentations

import gc
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
from sklearn.model_selection import train_test_split
%matplotlib inline

# Configuration
fold = 0
nfolds = 5
reduce = 4
sz = 256
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 1

# Constants
EPOCHS = 3
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

# Data loading
train_image, val_image,  = train_test_split(os.listdir(TRAIN_DIR), test_size=0.2, random_state=SEED)
val_image, test_image = train_test_split(val_image, test_size=0.5, random_state=SEED)

ds = ReadDataset(train_image, tfms=get_aug())
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
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1)
    return model

# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
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

cv_score = 0
test_score = 0
for fold in range(1):
    ds_t = ReadDataset(train_image, fold=fold, train=True, tfms=get_aug())
    ds_v = ReadDataset(val_image, fold=fold, train=False)
    dataloader_t = torch.utils.data.DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dataloader_v = torch.utils.data.DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = get_UnetPlusPlus().to(DEVICE)

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': model.encoder.parameters(), 'lr': 1e-4},
    ])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(dataloader_t))

    diceloss = DiceLoss()

    print(f"########FOLD: {fold}##############")

    for epoch in tqdm(range(EPOCHS)):
        # Train
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

        # Validation
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

    test_dataset = ReadDataset(test_image, fold=fold, train=False)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=NUM_WORKERS)
    test_loss = 0
    for data in dataloader_test:
        img, mask = data
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        outputs = model(img)

        loss = diceloss(outputs, mask)

        test_loss += loss.item()
    test_loss /= len(dataloader_test)

    print(f"FOLD: {fold}, EPOCH: {epoch + 1}, test_loss: {test_loss}")

    # Save model
    torch.save(model.state_dict(), f"FOLD{fold}_.pth")

    cv_score += valid_loss
    test_score += test_loss

cv_score = cv_score
test_score = test_score

# Visualization
# Load the trained model
model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
model.eval()

# Create a DataLoader for visualization (you can adjust batch_size and shuffle as needed)
visualization_dataloader = DataLoader(ds_v, batch_size=4, shuffle=False, num_workers=NUM_WORKERS)

# Visualization
with torch.no_grad():
    for i, data in enumerate(visualization_dataloader):
        img, mask = data
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        # Forward pass
        outputs = model(img)

        # Convert tensors to numpy arrays
        img_np = ((img.squeeze().permute(1, 2, 0).cpu().numpy() * std + mean) * 255.0).astype(np.uint8)
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = (outputs.squeeze().cpu().numpy() > TH).astype(np.uint8)

        # Plot the images, masks, and predictions
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
