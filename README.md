# Increasing Resolution and Accuracy in Sub-Seasonal Forecasting through 3D U-Net: the Western US

This repository provides code accompanying the paper
> Jihun Ryu, Hisu Kim, Shih-Yu (Simon) Wang, and Jin-Ho Yoon. (2025). Increasing Resolution and Accuracy in Sub-Seasonal Forecasting through 3D U-Net: the Western US

---
## Model Overview
This project employs a 3D U-Net architecture to enhance the accuracy and spatial resolution of sub-seasonal forecasts. The model is used as a post-processing method for Numerical Weather Prediction (NWP) outputs, particularly focusing on precipitation (pr) and temperature (t2m) forecasts.

---
## Data Availability
The data used in this study is available at the following links:
- [ECMWF](https://apps.ecmwf.int/datasets/data/s2s/levtype=sfc/type=pf/)
- [PRISM](https://prism.oregonstate.edu/)


---
## File Description
1. `code/`
    - `model_train.py`: Script to train a 3D U-Net model for sub-seasonal precipitation or temperature forecasting, with configurable training conditions via command-line arguments. *(For detailed argument options, refer to the script.)*
    - `model_test.py`: Script to perform inference using a trained 3D U-Net model for sub-seasonal precipitation or temperature forecasting, compute skill scores, and save the results. Similar to `model_train.py`, the test conditions are configurable via command-line arguments. *(For detailed argument options, refer to the script.)*
2. `model/`
    - `IRnAF.py`: Class of 3D U-Net model for sub-seasonal precipitation or temperature forecasting used in this study. It also includes model saving and training history management.
    - `evaluation.py`: Functions for skill score calculations, including RMSE, spatial correlation, and E-pre statistics, with latitude-based weighting for forecasts evaluation. 
3. `preprocess/`
    - `ECMWF_data.py`: Preprocesses ECMWF forecast data by downscaling from 1.5° to 0.25° resolution and standardizing ECMWF and PRISM data for consistency in model training.
    - `PRISM_data.py`: Upscales PRISM data to 0.25° resolution and stores it for analysis.


