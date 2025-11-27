#!/usr/bin/env python
#### Author: Jingwen Zhong, Date: 2025/07/30
"""
Evaluate the small transfer model trained on MERRA-2:
- Denormalizes predictions using norm_params.npz
- Compares against raw (log10-transformed) MERRA-2 truth
- Prints MAE and R^2 and writes metrics.json
"""
import os, re, json, argparse, numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from transfer_unet_fur_diffuser import get_area_data

N_LAG = 40

def infer_area_id(run_dir:str)->str:
    m = re.search(r'Area(\d+)', run_dir)
    if m: return m.group(1)
    return os.path.basename(run_dir)[-1]

def main(run_dir:str):
    y_hat = np.load(os.path.join(run_dir, "downscaled_mean.npy"))

    head = os.path.dirname(run_dir)
    test_days = np.load(os.path.join(head, "test_days.npy"))
    test_days = [int(d) for d in test_days if int(d) > N_LAG]

    area_id = infer_area_id(run_dir)
    m_data, *_ = get_area_data(area_id)
    y_true = m_data[np.array(test_days)-1]
    params = np.load(os.path.join(run_dir, "norm_params.npz"))
    m_min, m_max = float(params["m_min"]), float(params["m_max"])
    y_hat_denorm = y_hat * (m_max - m_min) + m_min

    assert y_hat_denorm.shape == y_true.shape, f"shape mismatch: {y_hat_denorm.shape} vs {y_true.shape}"

    mae = mean_absolute_error(y_true.reshape(-1), y_hat_denorm.reshape(-1))
    r2  = r2_score(y_true.reshape(-1), y_hat_denorm.reshape(-1))

    print(f"MAE (log10 units) = {mae:.6f}")
    print(f"R²                 = {r2:.6f}")

    with open(os.path.join(run_dir, "metrics_small.json"), "w") as f:
        json.dump({"mae_log10": float(mae), "r2": float(r2),
                   "n_pixels": int(y_true.size), "n_days": int(y_true.shape[0])}, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="e.g. .../Season2/Area1")
    main(**vars(p.parse_args()))
