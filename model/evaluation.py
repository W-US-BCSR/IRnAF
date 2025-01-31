import numpy as np

def skill_score_ensemble(da1, da2, lat):
    """
    Computes skill scores for each ensemble and returns arrays of
    correlation (scorr_list), RMSE (rmse_list), and E-pre (Epre_list).
    """
    scorr_list = []
    rmse_list = []
    Epre_list = []
    for i in range(da1.shape[1]):
        temp_scorr_list = []
        temp_rmse_list = []
        temp_Epre_list = []
        for j in range(da1.shape[0]):
            temp_scorr_list.append(scorr(da1[j,i], da2[j,i], lat))
            temp_rmse_list.append(rmses(da1[j,i], da2[j,i], lat))
            temp_Epre_list.append(Epre(da1[j,i], da2[j,i], lat))
        scorr_list.append(temp_scorr_list)
        rmse_list.append(temp_rmse_list)
        Epre_list.append(temp_Epre_list)
    return np.array(scorr_list), np.array(rmse_list), np.array(Epre_list)

def rmses(da1, da2, lat):
    """
    Calculates the RMSE (root mean square error) with latitude-based weighting.
    The weights are computed using cos(lat). 
    NaNs are masked out before computation.
    """
    # Calculate latitude-based weights
    weights = np.cos(np.deg2rad(lat))

    # Mask out NaNs
    mask = np.isnan(da1)
    da1 = np.ma.masked_where(mask, da1)
    da2 = np.ma.masked_where(mask, da2)

    # Compute mean of squared differences with weights
    diff_squared = (da1 - da2) ** 2
    mean_diff_squared = np.ma.average(diff_squared, weights=weights, axis=0)
    # Take the square root of the overall mean
    rmse = np.sqrt(mean_diff_squared.mean())
    return rmse

def scorr(da1, da2, lat):
    """
    Calculates the spatial correlation with latitude-based weighting.
    The weights are computed using cos(lat).
    NaNs are masked out before computation.
    """
    # Calculate latitude-based weights
    weights = np.cos(np.deg2rad(lat))

    # Mask out NaNs (both da1 and da2 at the same locations)
    mask = np.isnan(da1)
    da1 = np.ma.masked_where(mask, da1)
    da2 = np.ma.masked_where(mask, da2)

    # Compute weighted mean for da1 and da2
    da1_mean = np.ma.average(da1, weights=weights, axis=0)
    da1_mean = np.ma.average(da1_mean, axis=0)
    da2_mean = np.ma.average(da2, weights=weights, axis=0)
    da2_mean = np.ma.average(da2_mean, axis=0)

    # Compute anomalies
    da1_ano = da1 - np.expand_dims(da1_mean, axis=(0,1))
    da2_ano = da2 - np.expand_dims(da2_mean, axis=(0,1))

    # Compute weighted covariance
    cov = np.ma.average(da1_ano * da2_ano, weights=weights, axis=0).mean(axis=0)

    # Compute weighted standard deviations
    sd1 = np.sqrt(np.ma.average(da1_ano**2, weights=weights, axis=0).mean(axis=0))
    sd2 = np.sqrt(np.ma.average(da2_ano**2, weights=weights, axis=0).mean(axis=0))

    if sd1 == 0 or sd2 == 0:
        return np.nan

    # Correlation
    corr = cov / (sd1 * sd2)
    return corr

def Epre(da1, da2, lat):
    """
    Calculates the E-pre statistic with latitude-based weighting.
    The weights are computed using cos(lat).
    NaNs are masked out before computation.
    """
    # Calculate latitude-based weights
    weights = np.cos(np.deg2rad(lat))

    mask = np.isnan(da1)
    da1 = np.ma.masked_where(mask, da1)
    da2 = np.ma.masked_where(mask, da2)

    # Compute weighted mean for da1 and da2
    da1_mean = np.ma.average(da1, weights=weights, axis=0)
    da1_mean = np.ma.average(da1_mean, axis=0)
    da2_mean = np.ma.average(da2, weights=weights, axis=0)
    da2_mean = np.ma.average(da2_mean, axis=0)

    # Compute anomalies
    da1_ano = da1 - np.expand_dims(da1_mean, axis=(0,1))
    da2_ano = da2 - np.expand_dims(da2_mean, axis=(0,1))

    # Compute weighted covariance
    cov = np.ma.average(da1_ano * da2_ano, weights=weights, axis=0).mean(axis=0)

    # Compute weighted standard deviations
    sd1 = np.sqrt(np.ma.average(da1_ano**2, weights=weights, axis=0).mean(axis=0))
    sd2 = np.sqrt(np.ma.average(da2_ano**2, weights=weights, axis=0).mean(axis=0))

    if sd1 == 0 or sd2 == 0:
        return np.nan

    # Correlation
    corr = cov / (sd1 * sd2)

    # Evaluate the E-pre index
    E = np.log((sd1 / sd2 + sd2 / sd1)**2 * 2**4 / (1 + corr)**4)
    return E

def find_best_epoch(target: str, ens_name: str, var_num: int, is_mean: bool) -> int:
    """
    Reads the 'best_epoch' file to find the best epoch for a given combination of
    target, ens_name, var_num, and is_mean. If no matching record is found, raises an error.
    """
    with open('best_epoch', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            properties = line.split()
            if len(properties) == 5:
                if (str(properties[0]) == target and
                    str(properties[1]) == ens_name and
                    int(properties[2]) == var_num and
                    str(properties[3]) == str(is_mean)):
                    return int(properties[4])
    raise ValueError("Can't find the best epoch. Please check first that training has been done.")
