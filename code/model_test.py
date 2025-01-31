import os
import argparse
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model.IRnAF import UNet, CustomDataset, pearson_correlation_loss_with_weights
from model.evaluation import skill_score_ensemble

"""
This script loads the trained model (trained on precipitation or temperature),
performs inference (testing) on the test set, and then computes skill scores.
"""

parser = argparse.ArgumentParser(description='Run the model with different parameters')

# Arguments
parser.add_argument('--target', type=str, choices=['pr', 't2m'], required=True,
                    help='Target variable to predict (pr or t2m)')
parser.add_argument('--ens_num', type=int, choices=[1, 50], required=True,
                    help='Number of ensemble members (1 or 50)')
parser.add_argument('--var_num', type=int, choices=[1, 2, 4, 8], required=True,
                    help='Number of variables used as input to the model')
parser.add_argument('--is_mean', type=int, required=True,
                    help='If 1, use mean ensemble (AVG). Otherwise, raw ensemble.')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the saved model')
parser.add_argument('--file_path', type=str, required=True,
                    help='Path to the data files')
parser.add_argument('--best_epoch', type=int, required=True,
                    help='The best epoch to load the model weights from')

args = parser.parse_args()

# Parse arguments
target = args.target
ens_num = args.ens_num
var_num = args.var_num
is_mean = bool(args.is_mean)
model_path = args.model_path
file_path = args.file_path
best_epoch = args.best_epoch

# Determine ensemble name
if ens_num == 1:
    ens_name = '01'
else:
    ens_name = str(ens_num)

if is_mean:
    ens_name = 'AVG'
    ens_num = 1

# Define variables for each target
var_config = {
    "pr":  ['pr', 'mslp', 'z500', 'z200', 'u850', 'tcw', 'v500', 'v10'],
    "t2m": ['t2m', 'u500', 'z200', 'v200', 'z500', 'mslp', 'elevation', 'tcw']
}
vars = var_config[target][:var_num]
var_name = str(var_num)

data_label   = xr.open_dataarray("PRISM_tmean.nc")[0]

# Load test labels (shape: time x ens_num x lead_time x lat x lon)
test_label = np.load(f"{file_path}/PRISM_{target}_test_025_std_both_PRISM_NWP.npy")
test_label = test_label.transpose(1, 0, 2, 3, 4, 5)

# Prepare list to hold all test data arrays
test_da_list = []

# Helper function for loading each variable in the test set
def load_test_data(var_name, target, ens_num, file_path):
    """
    Loads test data for a single variable (var_name).
    If 'elevation', it is broadcasted for all times and ensembles.
    Different suffixes or paths are used depending on the target/variable.
    """
    if var_name == 'elevation':
        topo = np.load(f"{file_path}/{var_name}_025_std.npy").astype(np.float32)
        return np.broadcast_to(topo, (ens_num, 104, 1, 32, 72, 72))
    
    # For other variables, pick the right file suffix
    if target == 'pr':
        if var_name == 'pr':
            suffix = '_test_025_std_both_PRISM_NWP'
        elif var_name == 't2m':
            suffix = '_test_025_std_only_NWP'
        else:
            suffix = '_test_025_std'
    else:
        # target == 't2m'
        if var_name == 't2m':
            suffix = '_test_025_std_both_PRISM_NWP'
        elif var_name == 'pr':
            suffix = '_test_025_std_only_NWP'
        else:
            suffix = '_test_025_std'
    
    file_path = f"{file_path}/{var_name}{suffix}.npy"
    return np.load(file_path)

# Load test data for each variable
for v in vars:
    temp_test = load_test_data(v, 
                               target, 
                               ens_num, 
                               file_path)
    test_da_list.append(temp_test)

# Stack along new dimension for variables, remove any single dimension, and transpose
test_da = np.stack(test_da_list, axis=2)
test_da = test_da.squeeze(axis=3).transpose(1, 0, 2, 3, 4, 5)

# Slice lat/lon for North America region (modify lat/lon slice as needed)
na_lon = data_label.sel(lon=slice(235.5, 253.25), 
                        lat=slice(49, 31.25)).lon
na_lat = data_label.sel(lon=slice(235.5, 253.25), 
                        lat=slice(49, 31.25)).lat

# Create a dataset and dataloader for test set
TEST_BATCH_SIZE = 4
test_ds = CustomDataset(test_da, test_label)
test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False)

# Define the device for inference
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model and load the saved weights
unit = 64
model = UNet(ens_num, var_num, unit).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(model_path, f"model_epoch_{best_epoch+1}.pt")))
model.eval()  # Set model to evaluation mode

# Pre-calculated mean and std for re-scaling predictions
stats = {
    't2m': {
        'AVG': {'mean': 283.49344, 'std': 10.370236},
        '01':  {'mean': 283.4835,  'std': 10.604693},
        '50':  {'mean': 283.28806, 'std': 10.917166}
    },
    'pr': {
        'AVG': {'mean': 0.23615737, 'std': 0.2924946},
        '01':  {'mean': 0.18815486, 'std': 0.3233278},
        '50':  {'mean': 0.19532873, 'std': 0.31666538}
    }
}

mean_val = stats[target][ens_name]['mean']
std_val  = stats[target][ens_name]['std']

# Loss definitions
loss_function = nn.MSELoss()

# Inference variables
test_loss, test_mse, test_corr = 0, 0, 0
predictions = None
n_test_batches = len(test_loader)

# Inference loop
with torch.no_grad():
    for batch in test_loader:
        imgs = batch['imgs'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # Forward pass
        preds = model(imgs)

        # Calculate losses
        mse_loss = loss_function(preds, labels)
        corr_loss = pearson_correlation_loss_with_weights(preds, labels, na_lat)
        total_loss = mse_loss + corr_loss

        test_loss += total_loss.item()
        test_mse  += mse_loss.item()
        test_corr += corr_loss.item()

        # Convert predictions back to CPU NumPy
        preds_np = preds.detach().cpu().numpy()

        # Rescale predictions
        if target == 'pr':
            # For precipitation: inverse transform from log10(pr + 1)
            preds_rescaled = 10 ** ((preds_np.squeeze(axis=(1, 2)) * std_val) + mean_val) - 1
        else:
            # For temperature: just unstandardize
            preds_rescaled = preds_np.squeeze(axis=(1, 2)) * std_val + mean_val

        if predictions is None:
            predictions = preds_rescaled
        else:
            predictions = np.concatenate((predictions, preds_rescaled), axis=0)

# Averages over test batches
test_loss /= n_test_batches
test_mse  /= n_test_batches
test_corr /= n_test_batches

# Load reference data for skill score
ans = np.load(f"{file_path}/PRISM_{target}.npy").squeeze(axis=1)[835:939, :, :, :]

# Compute skill score over the NA region
score_na  = skill_score_ensemble(ans, predictions, na_lat)
score_na  = np.array(score_na)
na_mean   = np.mean(score_na, axis=2)
na_std    = np.std(score_na,  axis=2)
na_score  = np.stack([na_mean, na_std], axis=2)

# Compute difference (prediction - reference)
na_diff = predictions - ans

# Save final arrays (predictions and scores)
save_diff_file  = f"{target}_ENS{ens_name}_VAR{var_name}_diff.npy"
save_score_file = f"{target}_ENS{ens_name}_VAR{var_name}_score.npy"

np.save(save_diff_file,  na_diff)
np.save(save_score_file, na_score)

print(f"Test completed. \n"
      f"Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, Corr: {test_corr:.4f}\n"
      f"Results saved:\n {save_diff_file}\n {save_score_file}")
