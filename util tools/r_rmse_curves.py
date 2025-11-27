#### Author: Yang Xiang, Date: 2025.10.17
#### In-dataset temporal-lag diagnostics for G5NR and MERRA-2.
#### For each Season{1..4}/Area{0,1}, compute image-wise RMSE and R^2 between day t and (t - lag) for lags = 1..K 
#### within the SAME dataset, then plot the trend curves (G5NR vs MERRA-2). for each lag, the script calculates 
#### the metric for all possible pairs and then averages them.

import os, sys, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets"

if BASE not in sys.path:
    sys.path.append(BASE)
from train_ddpm_with_transfer import get_area_data
from util_tools.fixed_transfer_data_loader_old import data_processer

def _finite_rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b) 
    if not np.any(m): 
        return np.nan
    diff = a[m] - b[m]
    return float(np.sqrt(np.mean(diff * diff)))

def _finite_r2(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan
    r = np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1]
    return float(r * r) 

def compute_curves_for(season_dir: str, area_id: int, out_root: str, max_lag: int = 10):
    """
    season_dir: .../new_large_ddpm_3_with_transfer_result/SeasonX
    area_id: Area 0 or Area 1 
    out_root: to save csv and figures
    """
    area_dir = os.path.join(season_dir, f"Area{area_id}")
    os.makedirs(area_dir, exist_ok=True)
    train_days = np.load(os.path.join(season_dir, "train_days.npy"))
    test_days  = np.load(os.path.join(season_dir, "test_days.npy"))
    season_days = np.unique(np.sort(np.concatenate([train_days, test_days]))).tolist() 
    g5nr, merra, ele, [G_lats, G_lons], days_dummy, _ = get_area_data(area_id)
    assert g5nr.shape == merra.shape, "G5NR and MERRA-2 are misaligned in shape."
    T = g5nr.shape[0]
    assert T >= max(season_days), "Season day index out of bounds for loaded data."


    def per_lag_metrics(cube: np.ndarray, name: str):

        rows = []
        for lag in range(1, max_lag + 1):
            valid = [d for d in season_days if (d - lag) in season_days]
            if not valid:
                rows.append(dict(dataset=name, lag=lag,
                                 rmse_mean=np.nan, rmse_std=np.nan,
                                 r2_mean=np.nan, r2_std=np.nan, n_days=0))
                continue
            rmse_list, r2_list = [], []
            for d in valid:
                t  = d - 1     
                tk = d - lag - 1
                a  = cube[t]
                b  = cube[tk]
                rmse_list.append(_finite_rmse(a, b))
                r2_list.append(_finite_r2(a, b))

            rmse_arr = np.asarray(rmse_list, dtype=float)
            r2_arr   = np.asarray(r2_list, dtype=float)

            rows.append(dict(
                dataset=name, lag=lag, n_days=len(valid),
                rmse_mean=float(np.nanmean(rmse_arr)) if len(rmse_arr) else np.nan,
                rmse_std =float(np.nanstd (rmse_arr)) if len(rmse_arr) else np.nan,
                r2_mean  =float(np.nanmean(r2_arr))   if len(r2_arr)   else np.nan,
                r2_std   =float(np.nanstd (r2_arr))   if len(r2_arr)   else np.nan,
            ))
        return pd.DataFrame(rows)

    df_g = per_lag_metrics(g5nr,  "G5NR")
    df_m = per_lag_metrics(merra, "MERRA-2")
    df   = pd.concat([df_g, df_m], ignore_index=True)
    csv_out = os.path.join(out_root, f"in_dataset_lag_curves_area{area_id}.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved stats → {csv_out}")
    fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=180)
    for label, grp in df.groupby("dataset"):
        ax1.plot(grp["lag"], grp["rmse_mean"], marker="o", label=label)
    ax1.set_xlabel("lag (days)")
    ax1.set_ylabel("image-wise RMSE (log-AOD)")
    ax1.set_title(f"Season {os.path.basename(season_dir)} · Area{area_id} · RMSE vs lag")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    rmse_png = os.path.join(out_root, f"in_dataset_RMSE_vs_lag_Season{os.path.basename(season_dir)[-1]}_Area{area_id}.png")
    fig1.tight_layout(); fig1.savefig(rmse_png); plt.close(fig1)
    print(f"Saved figure → {rmse_png}")
    fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=180)
    for label, grp in df.groupby("dataset"):
        ax2.plot(grp["lag"], grp["r2_mean"], marker="o", label=label)
    ax2.set_xlabel("lag (days)")
    ax2.set_ylabel(r"$R^2$ (image-wise)")
    ax2.set_title(f"Season {os.path.basename(season_dir)} · Area{area_id} · R² vs lag")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    r2_png = os.path.join(out_root, f"in_dataset_R2_vs_lag_Season{os.path.basename(season_dir)[-1]}_Area{area_id}.png")
    fig2.tight_layout(); fig2.savefig(r2_png); plt.close(fig2)
    print(f"Saved figure → {r2_png}")

def main():
    ap = argparse.ArgumentParser(description="Compute in-dataset lag RMSE/R² curves for G5NR and MERRA-2.")
    ap.add_argument("run_root",
                    help="The training run root that holds Season*/ (e.g., .../new_large_ddpm_3_with_transfer_result)")
    ap.add_argument("--max_lag", type=int, default=10, help="maximum lag in days")
    args = ap.parse_args()

    run_root = os.path.abspath(args.run_root)
    for s in (1, 2, 3, 4):
        season_dir = os.path.join(run_root, f"Season{s}")
        if not os.path.isdir(season_dir):
            print(f"[WARN] missing {season_dir} – skipping")
            continue
        for area in (0, 1):
            print(f"\n▶ Season{s} / Area{area}")
            compute_curves_for(season_dir, area, out_root=season_dir, max_lag=args.max_lag)

if __name__ == "__main__":
    main()
