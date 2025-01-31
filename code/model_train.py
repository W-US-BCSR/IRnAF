import os
import argparse
import numpy as np
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.optim as optim

from model.IRnAF import *
from model.evaluation import *

"""
This script trains the model for either precipitation (pr) or
temperature (t2m) using PRISM data and NWP data.
"""

parser = argparse.ArgumentParser(description='Run the model with different parameters')

# 'target' argument (pr or t2m)
parser.add_argument('--target', type=str, choices=['pr', 't2m'], required=True,
                    help='Target variable to predict (pr or t2m)')

# 'ens_num' argument
parser.add_argument('--ens_num', type=int, required=True,
                    help='Number of ensemble members')

# 'var_num' argument (1, 2, 4, or 8)
parser.add_argument('--var_num', type=int, choices=[1, 2, 4, 8], required=True,
                    help='Number of variables used as input to the model')

# 'is_mean' argument (0 or 1)
parser.add_argument('--is_mean', type=int, required=True,
                    help='If set to 1, use mean ensemble (AVG). Otherwise, use raw ensemble.')

# 'model_save_path' argument (str)
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to save the trained model')

# 'file_path' argument (str)
parser.add_argument('--file_path', type=str, required=True,
                    help='Path to the data files')

args = parser.parse_args()

data_label   = xr.open_dataarray("PRISM_tmean.nc")[0]

target = args.target
ens_num = args.ens_num
var_num = args.var_num
is_mean = bool(args.is_mean)
model_save_path = args.model_path
file_path = args.file_path

if ens_num == 1:
    ens_name = '01'
else:
    ens_name = str(ens_num)

if is_mean:
    ens_name = 'AVG'
    ens_num = 1

var = {
    "pr":['pr', 'mslp', 'z500', 'z200', 'u850', 'tcw', 'v500', 'v10'],
    "t2m":['t2m', 'u500', 'z200', 'v200', 'z500', 'mslp', 'elevation', 'tcw']
}

vars = var[target][:var_num]

var_name = str(var_num)

train_label = np.load(f"{file_path}/PRISM_{target}_train_025_std_both_PRISM_NWP.npy").transpose(1, 0, 2, 3, 4, 5)
val_label = np.load(f"{file_path}/PRISM_{target}_val_025_std_both_PRISM_NWP.npy").transpose(1, 0, 2, 3, 4, 5)

def load_train_val(var, target, file_path, ens_num):
    if var == 'elevation':
        # For topo: broadcast the same topo array across all times, channels, etc.
        topo = np.load(f"{file_path}/{var}_025_std.npy").astype(np.float32)
        temp_train = np.broadcast_to(topo, (ens_num, 626, 1, 32, 72, 72))
        temp_val   = np.broadcast_to(topo, (ens_num, 104, 1, 32, 72, 72))
    else:
        # Decide on the file prefix/suffix
        if target == 'pr':
            if var == 'pr':
                suffix = '_025_std_both_PRISM_NWP'
            elif var == 't2m':
                suffix = '_025_std_only_NWP'
            else:
                suffix = '_025_std'
        else:
            # target == 't2m'
            if var == 't2m':
                suffix = '_025_std_both_PRISM_NWP'
            elif var == 'pr':
                suffix = '_025_std_only_NWP'
            else:
                suffix = '_025_std'
        
        # Load .npy files
        temp_train = np.load(f"{file_path}/{var}_train{suffix}.npy")
        temp_val   = np.load(f"{file_path}/{var}_val{suffix}.npy")

    return temp_train, temp_val

train_da, val_da = [], []

for var_name in vars:
    temp_train, temp_val = load_train_val(var_name, 
                                          target, 
                                          file_path, 
                                          ens_num)
    train_da.append(temp_train)
    val_da.append(temp_val)

    # Optionally delete temp_train if memory is a concern:
    del temp_train

# stack arrays (axis=2 -> variable dimension)
train_da = np.stack(train_da, axis=2)
val_da   = np.stack(val_da,   axis=2)

# remove single channel dimension, then transpose
train_da = train_da.squeeze(axis=3).transpose(1, 0, 2, 3, 4, 5)
val_da   = val_da.squeeze(axis=3).transpose(1, 0, 2, 3, 4, 5)

# subset of lat/lon for NA region
na_lon = data_label.sel(lon = slice(235.5, 253.25), 
                        lat = slice(49, 31.25)).lon
na_lat = data_label.sel(lon = slice(235.5, 253.25), 
                        lat = slice(49, 31.25)).lat

# training hyperparameters
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 11
N_EPOCHS = 100

# create datasets
train_ds = CustomDataset(train_da, train_label)
val_ds = CustomDataset(val_da, val_label)

# create data loaders
train_loader = torch.utils.data.DataLoader(train_ds, 
                                           batch_size=TRAIN_BATCH_SIZE, 
                                           shuffle=False
                                           )
val_loader = torch.utils.data.DataLoader(val_ds, 
                                         batch_size=TRAIN_BATCH_SIZE, 
                                         shuffle=False
                                         )

# flags for optional modules
USE_BATCH_NORMALIAZTION = True
USE_DROPOUT = True

# device info
cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"Using CUDA devices: {cuda_devices}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using Device:', DEVICE)

# model initialization
unit = 64
model = UNet(ens_num, var_num, unit).to(DEVICE)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []

def early_stopping(val_losses, patience=10, min_delta=0.001):
    if len(val_losses) < patience + 1:
        return False
    
    recent_losses = val_losses[-patience:]
    if min(recent_losses) > val_losses[-patience-1] + min_delta:
        return True
    return False

best_val_loss = float('inf')
best_model = None
best_epoch = 0

for epoch in range(N_EPOCHS):
    train_loss = train(model, 
                       train_loader, 
                       loss_function, 
                       optimizer, 
                       na_lat, 
                       epoch, 
                       N_EPOCHS, 
                       DEVICE
                       )
    
    val_loss = validation(model, 
                          val_loader, 
                          loss_function, 
                          na_lat, 
                          epoch, 
                          N_EPOCHS, 
                          DEVICE
                          )
    
    train_losses, val_losses = update_loss_acc(train_loss, 
                                               val_loss, 
                                               train_losses, 
                                               val_losses
                                               )

    train_history = {'train_losses': train_losses, 'val_losses': val_losses}
    
    # save model checkpoint
    save_result(model_save_path, train_history, epoch, model)

    print(f'epoch: [{epoch+1} / {N_EPOCHS}]')
    print(f'TRAIN: [loss: {train_loss[0]:.4f}, mse: {train_loss[1]:.4f}, corr: {train_loss[2]:.4f}]')
    print(f'VALIDATION: [loss: {val_loss[0]:.4f}, mse: {val_loss[1]:.4f}, corr: {val_loss[2]:.4f}] \n')

    # check if current val loss is the best
    if val_loss[0] < best_val_loss:
        best_val_loss = val_loss[0]
        best_epoch = epoch

    if early_stopping([loss[0] for loss in val_losses]):
        print(f"Training stopped early at epoch {epoch+1}")
        break


def log_best_epoch(target, ens_name, var_num, is_mean, best_epoch):
    with open('best_epoch', 'a') as f:
        f.write(f"{target} {ens_name} {var_num} {is_mean} {best_epoch}\n")

log_best_epoch(target, ens_name, var_num, is_mean, best_epoch)