import os
import math
import copy
import numpy as np
import xarray as xr
import pandas as pd
import glob
import time as tm
import itertools
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial
import random
import xesmf as xe

def single_level_convert_to_xarray(file, vars):
    var_temp = xr.open_dataset(file)
    var = var_temp[vars]

    lon = var.lon.values
    lat = var.lat.values
    times = var.time.values 
    ensemble = np.arange(1, 51) 

    var_values = var.values  
    if vars == 'sst' or vars == 'tcw' or vars == 'msl' or vars == 'tp':var_values= var_values[:, np.newaxis, :, :]
    var = xr.DataArray(var_values, coords=[ensemble, [times[0]], lat, lon], dims=["ensemble", "time", "lat", "lon"])
    return var

def multi_level_convert_to_xarray(file, vars):
    var_temp = xr.open_dataset(file)
    var = var_temp[vars]    
    var_list = list(var.coords.keys())

    lon = var.lon.values
    lat = var.lat.values
    times = var.time.values 
    ensemble = np.arange(1, 51) 
    if var_list.count('lev') == 1:
        lev = var.lev.values
    else:
        lev = var.plev.values

    var_values = var.values  
    var_values= var_values[:, np.newaxis, :, :, :]
    var = xr.DataArray(var_values, coords=[ensemble, [times[0]], lev, lat, lon], dims=["ensemble", "time", "lev", "lat", "lon"])
    return var

def process_and_combine(files, vars):
    if vars == 'gh' or vars == 'u' or vars == 'v':
        with mp.Pool() as pool:
            func = partial(multi_level_convert_to_xarray, vars=vars)
            da_list = pool.map(func, files)
    else:
        with mp.Pool() as pool:
            func = partial(single_level_convert_to_xarray, vars=vars)
            da_list = pool.map(func, files)

    combined_da = xr.concat(da_list, dim="time")
    return combined_da


def concat_single(arg):
    j, vars, file = arg
    temp = xr.open_dataarray(file)
    temp = temp.sel(lon = slice(220, 310), lat = slice(53, 20))
    target_grid = xr.Dataset({
        'lat': (['lat'], np.linspace(49.25, 24.75, 99)),
        'lon': (['lon'], np.linspace(235.5, 292.75, 230))
    })

    if vars == 'pr':
        temp = temp.where(temp >= 0, 0)
        regridder = xe.Regridder(temp, target_grid, 'conservative')
    if vars == 'tcw':
        regridder = xe.Regridder(temp, target_grid, 'conservative')
    else:
        regridder = xe.Regridder(temp, target_grid, 'bilinear')

    temp = regridder(temp)   
    return j, temp.values

def concat_multi(arg):
    j, levs, file = arg
    temp = xr.open_dataarray(file).sel(lev = levs, lon = slice(220, 310), lat = slice(53, 20))

    target_grid = xr.Dataset({
        'lat': (['lat'], np.linspace(49.25, 24.75, 99)),
        'lon': (['lon'], np.linspace(235.5, 292.75, 230))
    })
    
    regridder = xe.Regridder(temp, target_grid, 'bilinear')
    temp = regridder(temp)
    return j, temp.values


var_list = ['t2m', 'sst', 'tcw', 'mslp', 'u10', 'v10', 'pr', 'z', 'u', 'v']
varnm = ['2t', 'sst', 'tcw', 'msl', '10u', '10v', 'tp', 'gh', 'u', 'v']

for vars in range(len(var_list)):
    print(var_list[vars])
    for ld in range(1, 33):
        filepaths =glob.glob("input_path")
        outpaths = "output_path" ## ncfile

        filepaths.sort()
        da_merge = process_and_combine(filepaths, varnm[vars])

        da_merge.to_netcdf(outpaths)


vars = ['t2m', 'mslp', 'pr', 'tcw', 'u10', 'v10']
vars = ['u', 'v', 'z']
levs = [85000.,  50000., 20000.]
levs_str = ['850', '500', '200']

ensemble = 'ensemble'
time = 'initail_time'
lat = 'target_lat'
lon = 'target_lon'
LeadDays = np.arange(1,33)

for i in range(len(vars)):
    filepaths = glob.glob("raw_resolutions_data") ## ncfile
    filepaths.sort()

    temp_ens = {}
    temp_var = []

    if vars[i] == 'u' or vars[i] == 'v' or vars[i] == 'z':
        for k in range(len(levs)):
            temp_ens = {}
            temp_var = []
            with mp.Pool(processes=16) as pool:
                arg_list = [(j, levs[k], filepaths[j]) for j in range(len(filepaths))]
                result = pool.map(concat_multi, arg_list)
                temp_ens = {j: result[j] for j in range(len(result))}

            for key, value in temp_ens.items():
                temp_var.append(temp_ens[key][1])
            temp_var = np.stack(temp_var, axis=2)
            temp_var = xr.DataArray(temp_var, coords=[ensemble, time, LeadDays, lat, lon], dims=['ensemble', 'time', 'lead', 'lat', 'lon'])
            temp_var.to_netcdf(vars[i]+levs_str[k]+".nc")
            del temp_var
    else:
        with mp.Pool(processes=16) as pool:
            arg_list = [(j, vars[i], filepaths[j]) for j in range(len(filepaths))]
            result = pool.map(concat_single, arg_list)
            temp_ens = {j: result[j] for j in range(len(result))}

        for key, value in temp_ens.items():
            temp_var.append(temp_ens[key][1])
        temp_var = np.stack(temp_var, axis=2)
        temp_var = xr.DataArray(temp_var, coords=[ensemble, time, LeadDays, lat, lon], dims=['ensemble', 'time', 'lead', 'lat', 'lon'])
        temp_var.to_netcdf(vars[i]+".nc")
        del temp_var


vars = ['t2m', 'mslp', 'pr', 'tcw', 'u10', 'v10',
        'u850', 'u500', 'u200', 'v850', 'v500', 'v200',
        'z850', 'z500', 'z200']

for i in range(len(vars)):
    filepaths = glob.glob(vars[i]+".nc")
    temp = xr.open_dataarray(filepaths[0])
    temp = temp.sel(lon = slice(235.5, 253.25), lat = slice(49, 31.25)).values
    temp = temp[:,:,np.newaxis,:,:,:]
    np.save(vars[i]+".npy", temp)
    print(temp.shape)


ele_prism = np.load("PRISM_elevation.npy")

mean = np.nanmean(ele_prism)
std = np.nanstd(ele_prism)

ele_prism_std = (ele_prism - mean) / std
ele_prism_std = np.nan_to_num(ele_prism_std, nan=0.0)

np.save("elevation_025_std.npy", ele_prism_std)

times = xr.open_dataarray("t2m.nc").time
train_times = times[0:626]  # 2015- 2020
validation_times = times[643:747] # 2020/03 - 2021/02
test_times = times[835:939] # 2022
train_indices = np.arange(0, 626)          # 2015 - 2019
validation_indices = np.arange(643, 747)   # 2020/03 - 2021/02
test_indices = np.arange(835, 939)         # 2022
combined_indices = np.concatenate([train_indices, validation_indices, test_indices])


## Standardize Variables
vars = ['tcw', 'mslp', 'u10', 'v10', 'u850', 'u500', 'v850', 'v500', 'v200', 'z850', 'z500', 'z200']
ens_na = ['01', '50']
ens_nu = [1, 50]

for i in range(len(vars)):
    if vars[i] == 'tcw':
        temp = np.load(vars[i]+".npy")
        temp = np.log10(np.maximum(temp, 0) + 1)
        for j in range(len(ens_nu)):
            mean = temp[0:ens_nu[j], combined_indices].mean()
            std = temp[0:ens_nu[j], combined_indices].std()
            temp_ens = (temp[0:ens_nu[j]] - mean) / std

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std.npy", temp_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std.npy", temp_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std.npy", temp_ens[:,835:939])
    else:
        temp = np.load(vars[i]+".npy")
        for j in range(len(ens_nu)):
            mean = temp[0:ens_nu[j], combined_indices].mean()
            std = temp[0:ens_nu[j], combined_indices].std()
            temp_ens = (temp[0:ens_nu[j]] - mean) / std
    
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std.npy", temp_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std.npy", temp_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std.npy", temp_ens[:,835:939])

## Standardize Variables T2m & Pr
vars = ['t2m', 'pr']

for i in range(len(vars)):
    if vars[i] == 'pr':
        temp_label = np.load(vars[i]+".npy")
        temp_label = np.log10(np.maximum(temp_label, 0) + 1)[np.newaxis,:,:,:,:,:]
        temp_train = np.load(vars[i]+".npy")
        temp_train = np.log10(np.maximum(temp_train, 0) + 1)
        temp_tot = np.concatenate([temp_label, temp_train], axis=0)
        for j in range(len(ens_nu)):
            mean_only = np.nanmean(temp_train[0:ens_nu[j],combined_indices])
            std_only = np.nanstd(temp_train[0:ens_nu[j],combined_indices])
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_only) / std_only

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_only_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_only_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_only_NWP.npy", temp_train_ens[:,835:939])

            mean_both = np.nanmean(temp_tot[0:ens_nu[j]+1,combined_indices])
            std_both = np.nanstd(temp_tot[0:ens_nu[j]+1,combined_indices])
            temp_label_ens = (temp_label - mean_both) / std_both
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_both) / std_both
            temp_label_ens = np.nan_to_num(temp_label_ens, nan=0.0)

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_train_ens[:,835:939])

            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_label_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_label_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_label_ens[:,835:939])

            print(mean_both, std_both)

    else:
        temp_label = np.load("PRISM_"+vars[i]+".npy")[np.newaxis,:,:,:,:,:]
        temp_train = np.load(vars[i]+".npy")
        temp_tot = np.concatenate([temp_label, temp_train], axis=0)
        for j in range(len(ens_nu)):
            mean_only = np.nanmean(temp_train[0:ens_nu[j],combined_indices])
            std_only = np.nanstd(temp_train[0:ens_nu[j],combined_indices])
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_only) / std_only

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_only_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_only_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_only_NWP.npy", temp_train_ens[:,835:939])

            mean_both = np.nanmean(temp_tot[0:ens_nu[j]+1,combined_indices])
            std_both = np.nanstd(temp_tot[0:ens_nu[j]+1,combined_indices])
            temp_label_ens = (temp_label - mean_both) / std_both
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_both) / std_both
            temp_label_ens = np.nan_to_num(temp_label_ens, nan=0.0)

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_train_ens[:,835:939])

            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_label_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_label_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_label_ens[:,835:939])

            print(mean_both, std_both)

# Ensemble mean
## Standardize Variables
vars = ['tcw', 'mslp', 'u10', 'v10', 'u850', 'u500', 'v850', 'v500', 'v200', 'z850', 'z500', 'z200']
ens_na = ['AVG']
ens_nu = [1]

for i in range(len(vars)):
    if vars[i] == 'tcw':
        temp = np.load(vars[i]+".npy")
        temp = temp.mean(axis=0)[np.newaxis,:,:,:,:]
        temp = np.log10(np.maximum(temp, 0) + 1)
        for j in range(len(ens_nu)):
            mean = temp[0:ens_nu[j], combined_indices].mean()
            std = temp[0:ens_nu[j], combined_indices].std()
            temp_ens = (temp[0:ens_nu[j]] - mean) / std

    
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std.npy", temp_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std.npy", temp_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std.npy", temp_ens[:,835:939])
    else:
        temp = np.load(vars[i]+".npy")
        temp = temp.mean(axis=0)[np.newaxis,:,:,:,:]
        for j in range(len(ens_nu)):
            mean = temp[0:ens_nu[j], combined_indices].mean()
            std = temp[0:ens_nu[j], combined_indices].std()
            temp_ens = (temp[0:ens_nu[j]] - mean) / std

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std.npy", temp_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std.npy", temp_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std.npy", temp_ens[:,835:939])

## Standardize Variables T2m & Pr
vars = ['t2m', 'pr']

for i in range(len(vars)):
    if vars[i] == 'pr':
        temp_label = np.load("PRISM_"+vars[i]+".npy")
        temp_label = np.log10(np.maximum(temp_label, 0) + 1)[np.newaxis,:,:,:,:,:]
        temp_train = np.load(vars[i]+".npy")
        temp_train = temp_train.mean(axis=0)[np.newaxis,:,:,:,:]
        temp_train = np.log10(np.maximum(temp_train, 0) + 1)
        temp_tot = np.concatenate([temp_label, temp_train], axis=0)
        for j in range(len(ens_nu)):
            mean_only = np.nanmean(temp_train[0:ens_nu[j],combined_indices])
            std_only = np.nanstd(temp_train[0:ens_nu[j],combined_indices])
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_only) / std_only

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_only_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_only_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_only_NWP.npy", temp_train_ens[:,835:939])

            mean_both = np.nanmean(temp_tot[0:ens_nu[j]+1,combined_indices])
            std_both = np.nanstd(temp_tot[0:ens_nu[j]+1,combined_indices])
            temp_label_ens = (temp_label - mean_both) / std_both
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_both) / std_both
            temp_label_ens = np.nan_to_num(temp_label_ens, nan=0.0)

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_train_ens[:,835:939])

            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_label_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_label_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_label_ens[:,835:939])

            print(mean_both, std_both)

    else:
        temp_label = np.load("PRISM_"+vars[i]+".npy")[np.newaxis,:,:,:,:,:]
        temp_train = np.load(vars[i]+".npy")
        temp_train = temp_train.mean(axis=0)[np.newaxis,:,:,:,:]
        temp_tot = np.concatenate([temp_label, temp_train], axis=0)
        for j in range(len(ens_nu)):
            mean_only = np.nanmean(temp_train[0:ens_nu[j],combined_indices])
            std_only = np.nanstd(temp_train[0:ens_nu[j],combined_indices])
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_only) / std_only
    
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_only_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_only_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_only_NWP.npy", temp_train_ens[:,835:939])

            mean_both = np.nanmean(temp_tot[0:ens_nu[j]+1,combined_indices])
            std_both = np.nanstd(temp_tot[0:ens_nu[j]+1,combined_indices])
            temp_label_ens = (temp_label - mean_both) / std_both
            temp_train_ens = (temp_train[0:ens_nu[j]] - mean_both) / std_both
            temp_label_ens = np.nan_to_num(temp_label_ens, nan=0.0)

            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_train_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_train_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_train_ens[:,835:939])

            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_train_025_std_both_PRISM_NWP.npy", temp_label_ens[:,0:626])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_val_025_std_both_PRISM_NWP.npy", temp_label_ens[:,643:747])
            np.save("ENS"+ens_na[j]+"/PRISM_"+vars[i]+"_test_025_std_both_PRISM_NWP.npy", temp_label_ens[:,835:939])

            print(mean_both, std_both)
