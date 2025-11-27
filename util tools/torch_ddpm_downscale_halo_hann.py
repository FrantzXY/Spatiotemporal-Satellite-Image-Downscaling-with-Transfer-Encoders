###_______ Author: Yige Yan, Date: 2025-06-15 _______###
###### This is a special downscale script designed for DDPM in PyTorch and Diffusers packages. its _predict_low() is a new one. 


"""
3) How to regenerate (with the fixed script)
Because your training script imports the downscale module at process start, the running job won’t pick up your code change mid-run. After training finishes:

Delete stale inference/eval artifacts so your script doesn’t skip:

bash
rm -f /.../Season*/Area*/downscaled_mean_*.npy
rm -f /.../Season*/Area*/pred_days_*.npy
rm -f /.../Season*/results_area*_*\.csv
(Keep s2s_model.pt and norm_params.npz!)

Re-run the same driver for each Season/Area. Your __main__ logic will skip training (checkpoint exists) and run downscaling + evaluation with the updated downscale code:

bash
python -u $BASE/train_ddpm_with_transfer.py /.../SeasonX/AreaY
If you prefer to reuse the already-running process later, add this just before you build ds = torch_ddpm_downscale.downscaler(large):

python
import importlib; importlib.reload(torch_ddpm_downscale)
But since the job’s already in flight, the simpler path is the re-run above.
"""

import numpy as np
import sys
if '..' not in sys.path:
    sys.path.append('..')
from util_tools import fixed_transfer_data_loader_old as transfer_data_loader
from typing import Tuple
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


seed = 465

HALO = int(os.environ.get("HALO", 4))
USE_HANN = int(os.environ.get("USE_HANN", 1))
INFER_STEPS = int(os.environ.get("INFER_STEPS", "200"))

class downscaler():
    def __init__(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError(f"downscaler requires a PyTorch nn.Module, got {type(model)}")
        self.model = model
        from diffusers import DDPMScheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=2000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        assert self.noise_scheduler.config.prediction_type == "sample", "Inference must be x0-mode"
        assert not self.noise_scheduler.config.clip_sample, "clip_sample must be False in x0-mode"

        self.task_dim = (16, 16)
        self.rng = np.random.default_rng(seed)
        self.is_torch = isinstance(model, nn.Module)

        if self.is_torch:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
            any_q = next(
                p
                for n, p in self.model.named_parameters()
                if ".to_q.weight" in n
            )
            heads_detected = any_q.shape[0] // 64
            gpu_limit = 4096
            n_heads = heads_detected
            self.max_batch = min(240, (gpu_limit - 1) // n_heads)

            def _predict_low(x_high_np, x_low_np, x_ele_np, x_misc_np):
                """
                DDPM inference with safe batching (x0 prediction).
                """
                cfg = self.noise_scheduler.config
                assert cfg.prediction_type == "sample" and not cfg.clip_sample
                B, H, W = x_high_np.shape[0], *self.task_dim
                outs = []

                self.noise_scheduler.set_timesteps(INFER_STEPS, device=self.device)

                for s in range(0, B, self.max_batch):
                    e = min(s + self.max_batch, B)

                    high = torch.from_numpy(x_high_np[s:e]).float().to(self.device)
                    low = torch.from_numpy(x_low_np[s:e]).float().to(self.device)
                    if low.ndim == 3:
                        low = low.unsqueeze(1)
                    ele = torch.from_numpy(x_ele_np[s:e]).float().to(self.device)
                    misc = (
                        torch.from_numpy(x_misc_np[s:e])
                        .float()
                        .to(self.device)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand(-1, -1, H, W)
                    )

                    x = torch.randn((e - s, 1, H, W), device=self.device)

                    for t in self.noise_scheduler.timesteps:
                        ts = torch.full((e - s,), t, device=self.device, dtype=torch.long)
                        with torch.no_grad():
                            pred_x0 = self.model(x, high, low, ele, misc, ts)

                        step_out = self.noise_scheduler.step(
                            model_output=pred_x0,
                            timestep=t,
                            sample=x,
                        )
                        x = step_out.prev_sample

                    outs.append(x.cpu())

                return torch.cat(outs, 0).numpy()

            self._predict_low = _predict_low

            def _predict_small_low(x_low_np: np.ndarray) -> np.ndarray:
                x_np = x_low_np[..., 0]
                with torch.no_grad():
                    x = torch.from_numpy(x_np).float().to(self.device)
                    out = self.model(x)
                return out.cpu().numpy()

            self._predict_small_low = _predict_small_low

    def downscale(
        self,
        h_data,
        l_data,
        ele_data,
        lat_lon,
        days,
        n_lag,
        n_pred,
        task_dim,
        stride,
        n_est=100,
        min_value=0,
    ):
        '''
        At each step we will feed the model a cube of data shaped
        (n_lag times, task_lat_dim rows, task_lon_dim cols)
        and ask it to predict the next n_pred times.

        :param h_data: (high resolution data, G5NR) It only contains data before the period we want to predict.
        :param l_data: range same as downscaled (low resolution data, MERRA2, has  the entire period we want to downscale)
        :param ele_data: fixed
        :param lat_lon: fixed
        :param days: range same as l_data
        :param days: range same as l_data
        :param n_lag: (input to the function flatten() in transfer_data_loader.py)
        :param n_pred: (input to the function flatten() in transfer_data_loader.py)
        :param task_dim: (input to the function flatten() in tarnsfer_data_loader.py)
        :param n_est=100 (the number of estimations)
        :param min_value=0: A physical constraint to prevent the model from predicting unrealistic AOD values.
        :return:
        '''
        G_lats, G_lons, M_lats, M_lons = lat_lon
        if M_lats is None or M_lons is None:
            M_lats, M_lons = G_lats, G_lons

        data_processor = transfer_data_loader.data_processer()
        if l_data.shape[1:] != h_data.shape[1:]:
            l_data = data_processor.unify_m_data(h_data, l_data, G_lats, G_lons, M_lats, M_lons)

        temp_mean_matrix = np.zeros((n_pred, n_pred, h_data.shape[1], h_data.shape[2]))

        for i in range(l_data.shape[0] - n_lag - n_pred + 1):
            l_win = l_data[i : i + n_lag + 1]
            days_win = days[i : i + n_lag + 1]
            X_high, X_low, X_ele, X_other = data_processor.flatten(
                h_data[-n_lag:],
                l_win,
                ele_data,
                [G_lats, G_lons],
                days_win,
                n_lag=n_lag,
                n_pred=n_pred,
                task_dim=task_dim,
                is_perm=False,
                return_Y=False,
                stride=stride,
            )

            list_of_predictions = []

            X_high = np.transpose(X_high, (0, 1, 4, 2, 3))
            X_low = X_low[..., 0]
            X_ele = X_ele[..., 0][:, None, :, :]
            X_misc = X_other[:, [0, 1, 3]]

            for _ in range(n_est):
                temp = self._predict_low(X_high, X_low, X_ele, X_misc)

                if isinstance(temp, torch.Tensor):
                    temp = temp.cpu().numpy()
                temp[temp < min_value] = (
                    min_value
                    + self.rng.random(size=temp[temp < min_value].shape) * 0.1
                )
                list_of_predictions.append(temp)

            pred_ensemble = np.stack(list_of_predictions, axis=1)
            pred_mean = np.mean(pred_ensemble, axis=1)

            pred_mean_list = self._reconstruct(
                pred_mean, h_data.shape[1:], task_dim=task_dim, stride=stride
            )

            temp_mean_matrix = np.concatenate(
                [
                    temp_mean_matrix[1:],
                    np.expand_dims(pred_mean_list, 0),
                ],
                axis=0,
            )
            current_mean_est = np.mean(
                np.array([temp_mean_matrix[k, n_pred - k - 1] for k in range(n_pred)]),
                axis=0,
            )
            h_data = np.concatenate(
                [h_data, np.expand_dims(current_mean_est, 0)],
                axis=0,
            )

        num_predictions_made = l_data.shape[0] - n_lag - n_pred + 1

        return h_data[-num_predictions_made:]

    def transfer_downscale(
        self,
        l_data_initial: np.ndarray,
        M_lats: np.ndarray,
        M_lons: np.ndarray,
        n_lag: int,
        n_pred: int,
        task_dim: tuple,
        *,
        n_est: int = 1,
        stride: int = 1,
        min_value: float = 0,
    ) -> np.ndarray:
        """
        Predicts one day ahead, then feeds that prediction back in
        autoregressively until the end of the provided time span.

        Returns
        -------
        np.ndarray  shape = (T_out, H, W)
            High-res estimates for every day *after* the initial n_lag window.
        """
        proc = transfer_data_loader.data_processer()
        l_data = l_data_initial.copy()
        out_frames = []

        rng = np.random.default_rng(seed)

        n_steps = l_data.shape[0] - n_lag

        for step in range(n_steps):
            cur_cube = l_data[step : step + n_lag]
            X_low = proc.flatten_transfer(
                cur_cube,
                n_lag=n_lag,
                n_pred=n_pred,
                task_dim=task_dim,
                is_perm=False,
                return_Y=False,
                stride=stride,
            )

            mc_stack = []
            for _ in range(n_est):
                y_hat = self._predict_small_low(X_low)
                bad = y_hat < min_value
                if bad.any():
                    y_hat[bad] = min_value + rng.random(size=bad.sum()) * 0.1
                mc_stack.append(y_hat)

            mean_patch = np.mean(np.stack(mc_stack, axis=1), axis=1)

            full_img = self._reconstruct(
                mean_patch,
                (len(M_lats), len(M_lons)),
                task_dim,
                stride=stride,
            )[0]

            out_frames.append(full_img)

            l_data = np.concatenate([l_data, full_img[None, ...]], axis=0)

        return np.asarray(out_frames, dtype=np.float32)

    def _reconstruct(self, pred_Y, org_dim, task_dim, stride):
        '''
        Reconstructs a set of small image patches back into a series of large images.
        :param pred_Y: Input tensor of patches with shape (num_patches, n_pred, patch_height, patch_width).
        X_high.shape[0] is how many patches you extracted with flatten(),
        which is n_instance = (h_data.shape[0]-n_lag-n_pred+1)*(h_data.shape[1]-task_lat_dim+1)*(h_data.shape[2]-task_lon_dim+1)
        This is the predicted sub-batches images by the generator!
        :param org_dim: A tuple (height, width) for the final reconstructed image.
        :param task_dim: A tuple (patch_height, patch_width) describing the patch size.
        :return: A tensor of reconstructed images with shape (n_pred, original G5NR lat height, original G5NR lon width).
        '''
        num_patches, n_pred, patch_height, patch_width = pred_Y.shape

        eps = 1e-6
        if USE_HANN:
            wx = np.hanning(patch_width).astype(np.float32)
            wy = np.hanning(patch_height).astype(np.float32)
            w2d = (wy[:, None] * wx[None, :]).astype(np.float32)
        else:
            w2d = np.ones((patch_height, patch_width), dtype=np.float32)
        w2d[w2d < eps] = eps

        sum_mtx = np.zeros((n_pred, org_dim[0], org_dim[1]), dtype=np.float32)
        weight_mtx = np.zeros((n_pred, org_dim[0], org_dim[1]), dtype=np.float32)

        patches_per_row = math.ceil((org_dim[1] - patch_width) / stride) + 1

        for i in range(num_patches):
            current_patch_data = pred_Y[i].astype(np.float32)

            row_index = i // patches_per_row
            col_index = i % patches_per_row

            lat_start = min(row_index * stride, org_dim[0] - patch_height)
            lon_start = min(col_index * stride, org_dim[1] - patch_width)

            h_in0 = HALO if lat_start > 0 else 0
            w_in0 = HALO if lon_start > 0 else 0
            h_in1 = patch_height - (HALO if (lat_start + patch_height) < org_dim[0] else 0)
            w_in1 = patch_width - (HALO if (lon_start + patch_width) < org_dim[1] else 0)

            if h_in1 <= h_in0 or w_in1 <= w_in0:
                h_in0 = 0
                w_in0 = 0
                h_in1 = patch_height
                w_in1 = patch_width

            sl_y = slice(lat_start + h_in0, lat_start + h_in1)
            sl_x = slice(lon_start + w_in0, lon_start + w_in1)

            patch_core = current_patch_data[:, h_in0:h_in1, w_in0:w_in1]
            w_core = w2d[h_in0:h_in1, w_in0:w_in1]

            sum_mtx[:, sl_y, sl_x] += patch_core * w_core[None, :, :]
            weight_mtx[:, sl_y, sl_x] += w_core[None, :, :]

        weight_mtx[weight_mtx < eps] = eps
        out = sum_mtx / weight_mtx

        return out
