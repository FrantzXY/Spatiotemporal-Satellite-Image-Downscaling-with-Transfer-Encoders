#### NOTE This script is strictly for the DDPM evaluation step. 
#### We use the small model in /project/def-mere/merra2/Ace_Transfer_Downscale/large_model
#### Author: Yang Xiang, Date: 2025/07/31


import os, sys, json, numpy as np, pandas as pd
from typing import List
from scipy.stats import linregress

def denormalize(data, min_val, max_val):
    """Scales G5NR data from normalized range [0, 1] back to its original range of AOD values."""
    return data * (max_val - min_val) + min_val

def nse(obs, sim):   
    """Nash-Sutcliffe Efficiency"""
    num = np.nansum((sim - obs) ** 2)
    den = np.nansum((obs - np.nanmean(obs)) ** 2)
    return 1 - num / den if den != 0 else np.nan

def kge(obs, sim):
    """Kling-Gupta Efficiency"""
    mean_o, std_o = np.nanmean(obs), np.nanstd(obs)
    mean_s, std_s = np.nanmean(sim), np.nanstd(sim)
    r   = np.corrcoef(obs.ravel(), sim.ravel())[0,1]
    beta= mean_s / mean_o if mean_o else np.nan
    gamma = std_s / std_o if std_o else np.nan
    return 1 - np.sqrt( (r-1)**2 + (beta-1)**2 + (gamma-1)**2 )

def rsq(obs, sim):
    """Coefficient of determination on finite pixels."""
    m = np.isfinite(obs) & np.isfinite(sim)

    if m.sum() < 2: return np.nan
    return np.corrcoef(obs[m], sim[m])[0,1]**2


if len(sys.argv) != 2:
    print("Usage:  python eval_large_unet.py  /.../SeasonX/AreaY")
    sys.exit(1)  

area_dir   = os.path.abspath(sys.argv[1])              
season_dir = os.path.dirname(area_dir)                   
area_id    = int(os.path.basename(area_dir)[-1])


modes = ["strict"]
for mode in modes:  
    csv_path = os.path.join(season_dir, f"results_area{area_id}_{mode}.csv")
    if os.path.exists(csv_path):
        print(f"✓ {mode} metrics already exist – skipping.")
        continue
    
    print(f"\n=== {mode.upper()} evaluation ===")

    y_hat_normalized = np.load(os.path.join(area_dir, f"downscaled_mean_{mode}.npy"))

    test_days = np.load(os.path.join(area_dir, f"pred_days_{mode}.npy"))
    print("✓ using pred_days.npy (", len(test_days), "days )")
    
    if len(y_hat_normalized) != len(test_days):
        raise RuntimeError(f"Prediction count {len(y_hat_normalized)} ≠ day list {len(test_days)}")
    AHEAD = "/project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets"
    if AHEAD not in sys.path:
        sys.path.append(AHEAD)

    from train_ddpm_with_transfer import get_area_data 
    g_data, _, _, _, _, _ = get_area_data(area_id)      
    y_true = g_data[np.array(test_days) - 1]           

    norm_params_path = os.path.join(area_dir, 'norm_params.npz')
    params = np.load(norm_params_path)
    g_min, g_max = params['g_min'], params['g_max']

    
    y_hat = denormalize(y_hat_normalized, g_min, g_max)

    assert y_true.shape == y_hat.shape, "Log transformed truth and log transformed prediction shapes mismatch"

    rmse = float(np.sqrt(np.nanmean((y_hat - y_true) ** 2)))
    mae  = float(np.nanmean(np.abs(y_hat - y_true)))
    r2   = float(rsq(y_true, y_hat))
    nse_ = float(nse(y_true, y_hat))
    kge_ = float(kge(y_true, y_hat))

    print(f"Here is the overall result: RMSE={rmse:.4f}  MAE={mae:.4f} R²={r2:.4f}  NSE={nse_:.4f}  KGE={kge_:.4f}")

    rows = []
    for d, t_img, p_img in zip(test_days, y_true, y_hat):
        rows.append(dict(
            day=int(d),
            rmse=float(np.sqrt(np.nanmean((p_img-t_img)**2))),
            mae=float(np.nanmean(np.abs(p_img-t_img))),
            r2=float(rsq(t_img, p_img)),
            nse=float(nse(t_img, p_img)),
            kge=float(kge(t_img, p_img)),
        ))
    df = pd.DataFrame(rows)
    overall = pd.DataFrame([dict(day=-999, rmse=rmse, mae=mae, r2=r2, nse=nse_, kge=kge_)])

    csv_path = os.path.join(season_dir, f"results_area{area_id}_{mode}.csv")
    df.to_csv(csv_path, index=False)
    print("Saved per-day metrics →", csv_path)
