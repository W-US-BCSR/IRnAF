import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam

from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'imgs': torch.tensor(self.imgs[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        return sample

class UNet(nn.Module):
    def __init__(self, ens_num, var_num, unit = 64):
        super().__init__()
        self.ens = ens_num
        self.var = var_num
        self.unit = unit

        def CBR3d(input_channel, output_channel, kernel_size=3, stride=1, padding=1):
            layer = nn.Sequential(
                nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(output_channel),
                nn.GELU()
            )
            return layer
        
        # Contracting path
        self.conv1 = nn.Sequential(
            CBR3d(ens_num * var_num, unit),
            CBR3d(unit, unit)
        )
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            CBR3d(unit, unit * 2),
            CBR3d(unit * 2, unit * 2)
        )
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            CBR3d(unit * 2, unit * 4),
            CBR3d(unit * 4, unit * 4)
        )
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleNeck = nn.Sequential(
            CBR3d(unit * 4, unit * 8),
            CBR3d(unit * 8, unit * 8),
        )
     
        # Expanding path
        self.upconv1 = nn.ConvTranspose3d(in_channels=unit * 8, out_channels=unit * 4, kernel_size=2, stride=2)
        self.ex_conv1 = nn.Sequential(
            CBR3d(unit * 8, unit * 4),
            CBR3d(unit * 4, unit * 4)
        )

        self.upconv2 = nn.ConvTranspose3d(in_channels=unit * 4, out_channels=unit * 2, kernel_size=2, stride=2)
        self.ex_conv2 = nn.Sequential(
            CBR3d(unit * 4, unit * 2),
            CBR3d(unit * 2, unit * 2)
        )

        self.upconv3 = nn.ConvTranspose3d(in_channels=unit * 2, out_channels=unit, kernel_size=2, stride=2)
        self.ex_conv3 = nn.Sequential(
            CBR3d(unit * 2, unit),
            CBR3d(unit, unit),
        )

        # Final conv layer to adjust the output channels and dimensions
        self.fc = nn.Conv3d(unit, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch, ens, var, time, lat, lon = x.size()
        x = x.view(batch, ens * var, time, lat, lon)        
        # Contracting path
        layer1 = self.conv1(x)
        out = self.pool1(layer1)

        layer2 = self.conv2(out)
        out = self.pool2(layer2)

        layer3 = self.conv3(out)
        out = self.pool3(layer3)

        # Bottleneck
        bottleNeck = self.bottleNeck(out)

        # Expanding path
        upconv1 = self.upconv1(bottleNeck)
        cat1 = torch.cat((layer3, upconv1), dim=1)
        ex_layer1 = self.ex_conv1(cat1)

        upconv2 = self.upconv2(ex_layer1)
        cat2 = torch.cat((layer2, upconv2), dim=1)
        ex_layer2 = self.ex_conv2(cat2)

        upconv3 = self.upconv3(ex_layer2)
        cat3 = torch.cat((layer1, upconv3), dim=1)
        ex_layer3 = self.ex_conv3(cat3)

        out = self.fc(ex_layer3)
        out = out.view(batch, 1, 1, time, lat, lon)
        return out
    
   

def pearson_correlation_loss_with_weights(output, target, lat):

    output = output.squeeze(1).squeeze(1)  # (batch, time, lon, lat)
    target = target.squeeze(1).squeeze(1)  # (batch, time, lon, lat)
    
    weights = np.cos(np.deg2rad(lat.values))
    weights = torch.tensor(weights, dtype=torch.float32).to(output.device)
    
    weights = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, lat)
    
    batch_size, time_size, lon_size, lat_size = output.shape
    losses = []
    
    for b in range(batch_size):
        for t in range(time_size):
            out_b_t = output[b, t, :, :]
            tar_b_t = target[b, t, :, :]
            
            out_centered = out_b_t - torch.mean(out_b_t)
            tar_centered = tar_b_t - torch.mean(tar_b_t)
            
            weighted_out = out_centered * weights
            weighted_tar = tar_centered * weights
            
            covariance = torch.sum(weighted_out * weighted_tar)
            variance_out = torch.sqrt(torch.sum(weighted_out ** 2))
            variance_tar = torch.sqrt(torch.sum(weighted_tar ** 2))
            
            correlation = covariance / (variance_out * variance_tar)
            
            loss = 1 - correlation
            losses.append(loss)
    
    return torch.mean(torch.stack(losses))

# Unet Training, validation, save Definition
def train(model, train_loader, loss_function, optimizer, lat, epoch, N_EPOCHS, DEVICE):
    model.train()
    train_loss = 0
    train_mse = 0
    train_corr = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
        imgs = batch['imgs'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        preds = model(imgs)
        mse_loss = loss_function(preds, labels)
        corr_loss = pearson_correlation_loss_with_weights(preds, labels, lat)

        loss = mse_loss + corr_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mse += mse_loss.item()
        train_corr += corr_loss.item()

    train_loss /= len(train_loader)
    train_mse /= len(train_loader)
    train_corr /= len(train_loader)

    return train_loss, train_mse, train_corr

def validation(model, val_loader, loss_function, lat, epoch, N_EPOCHS, DEVICE):
    val_loss = 0
    val_mse = 0
    val_corr = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}"):
            imgs = batch['imgs'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            preds = model(imgs)
            mse_loss = loss_function(preds, labels)
            corr_loss = pearson_correlation_loss_with_weights(preds, labels, lat)
            
            loss = mse_loss + corr_loss

            val_loss += loss.item()
            val_mse += mse_loss.item()
            val_corr += corr_loss.item()

    val_loss /= len(val_loader)
    val_mse /= len(val_loader)
    val_corr /= len(val_loader)
    
    return val_loss, val_mse, val_corr

def update_loss_acc(train_loss, val_loss, train_losses, val_losses):
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    return train_losses, val_losses

def save_result(save_path, train_history, epoch, model):
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
            print(f"Directory created: {save_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return 
    else:
        print(f"Directory already exists: {save_path}")

    save_model_file = os.path.join(save_path, f'model_epoch_{epoch+1}.pt')
    torch.save(model.state_dict(), save_model_file)

    save_pkl_file = os.path.join(save_path, 'train_history.pkl')
    with open(save_pkl_file, 'wb') as f:
        pickle.dump(train_history, f)

