import xarray as xr
import rasterio
import numpy as np
from glob import glob
import os
import multiprocessing as mp
from datetime import datetime
import xesmf as xe

def convert_to_xarray(file):
    with rasterio.open(file) as dataset:
        data = dataset.read(1)
        data = np.where(data == -9999, np.nan, data)
        transform = (0.041666666666666664, 0.0, -125.02083333333333, 0.0, -0.041666666666666664, 49.93750000000001)
        width = dataset.width
        height = dataset.height
        nodata = dataset.nodata

        lon_coords = np.arange(width) * transform[0] + transform[2]
        lat_coords = np.arange(height) * transform[4] + transform[5]

        filename = os.path.basename(file)
        date_str = filename.split('_')[4]
        date = datetime.strptime(date_str, '%Y%m%d')

        da = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={
                "lat": lat_coords,
                "lon": lon_coords,
                "time": date
            },
        )
        return da

def process_and_combine(files):
    with mp.Pool() as pool:
        da_list = pool.map(convert_to_xarray, files)
    combined_da = xr.concat(da_list, dim="time")
    return combined_da

# Temperature
path_t_mean = 'Input_path'
save_path = 'Save_path'
output_file = "PRISM_tmean.nc"

yr = range(2015,2025)
filepaths = []
for i in yr:
    file_pattern = f'{path_t_mean}/{i}/PRISM_tmean_stable_4kmD2_*.bil'
    files = sorted(glob(file_pattern))
    filepaths = filepaths + files

combined_ds = process_and_combine(filepaths)
combined_ds = combined_ds + 273.15
combined_ds['lon'] = xr.where(combined_ds['lon'] < 0, combined_ds['lon'] + 360, combined_ds['lon'])
combined_ds.to_netcdf(f"{save_path}/{output_file}")

# Precipitation
path_precip = 'Input_path'
save_path = 'Save_path'
output_file = "PRISM_precip.nc"

yr = range(2015,2025)
filepaths = []
for i in yr:
    file_pattern = f'{path_precip}/{i}/PRISM_ppt_stable_4kmD2_*.bil'
    files = sorted(glob(file_pattern))
    filepaths = filepaths + files

combined_ds = process_and_combine(filepaths)
combined_ds['lon'] = xr.where(combined_ds['lon'] < 0, combined_ds['lon'] + 360, combined_ds['lon'])
combined_ds.to_netcdf(f"{save_path}/{output_file}")

# Elevation
path_ele = 'Input_path'
save_path = 'Save_path'
output_file = "PRISM_elevation.nc"

with rasterio.open(path_ele) as dataset:
    data = dataset.read(1)
    data = np.where(data == -9999, np.nan, data)
    transform = dataset.transform
    transform = (0.041666666666666664, 0.0, -125.02083333333333, 0.0, -0.041666666666666664, 49.93750000000001)
    crs = dataset.crs
    width = dataset.width
    height = dataset.height
    nodata = dataset.nodata
    
    lon_coords = np.arange(width) * transform[0] + transform[2]
    lat_coords = np.arange(height) * transform[4] + transform[5]

    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lat_coords, "lon": lon_coords},
    )

da['lon'] = xr.where(da['lon'] < 0, da['lon'] + 360, da['lon'])
da.to_netcdf(f"{save_path}/{output_file}")

# Regrid and Sel data
t2m_data = xr.open_dataarray('t2m_save_path')
pr_data = xr.open_dataarray('pr_save_path')
ele_data = xr.open_dataarray('ele_save_path')

target_grid = xr.Dataset({
        'lat': (['lat'], np.linspace(49.25, 24.75, 99)),
        'lon': (['lon'], np.linspace(235.5, 292.75, 230))
})

t2m_regridder = xe.Regridder(t2m_data, target_grid, 'bilinear')
pr_regridder = xe.Regridder(pr_data, target_grid, 'conservative')
ele_regridder = xe.Regridder(ele_data, target_grid, 'bilinear')

t2m_regrid = t2m_regridder(t2m_data)
pr_regrid = pr_regridder(pr_data)
ele_regrid = ele_regridder(ele_data)

t2m_data = t2m_regrid.sel(lon = slice(235.5, 253.25), lat = slice(49, 31.25)).values
pr_data = pr_regrid.sel(lon = slice(235.5, 253.25), lat = slice(49, 31.25)).values
ele_data = ele_regrid.sel(lon = slice(235.5, 253.25), lat = slice(49, 31.25)).values

np.save('outpath/PRISM_elevation.npy', ele_data)

vars = ['t2m', 'pr']

for i in range(len(vars)):
    filepaths = ['initial_time_list']
    globals()[vars[i]+'_temp'] = []
    for j in range(len(filepaths)):
        times = 'lead_time_per_initial_time' 
        temp = globals()[vars[i]+'_prism'].sel(time=times)
        globals()[vars[i]+'_temp'].append(temp)

    globals()[vars[i]+'_temp'] = np.stack(globals()[vars[i]+'_temp'], axis=1)[:,np.newaxis,:,:,:]

    np.save("outpath/PRISM_"+vars[i]+".npy", globals()[vars[i]+'_temp'])