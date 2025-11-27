#### This is a script of a large generative Denoising Diffusion Proabilistic Model (DDPM) with small transfer model for image downscaling.
#### It is a slight variant of J Ho's original DDPM (https://arxiv.org/abs/2006.11239)
#### We let the model to predict the clean image X0 at t0 rather than predicting the Gaussian noises in the reverse process.   
#### We use the small model in /project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets/small_transfer_model/new_small_result
#### Author: Yang Xiang, Date: 2025/07/31

import numpy as np
import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import diffusers
from diffusers import UNet2DModel
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers import DDPMScheduler
import pandas as pd

seed = 465
np.random.seed(seed)
torch.manual_seed(seed)

TILE = (16, 16)
HALO = int(os.environ.get("HALO", "4"))
STRIDE_TRAIN = int(os.environ.get("STRIDE_TRAIN", "8"))
STRIDE_INFER = int(os.environ.get("STRIDE_INFER", "2"))
USE_HANN = int(os.environ.get("USE_HANN", "1"))
INFER_STEPS = int(os.environ.get("INFER_STEPS", "1000"))
N_EST = int(os.environ.get("N_EST", "1"))

AHEAD = "/project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets"
if AHEAD not in sys.path:
    sys.path.append(AHEAD)
from util_tools.fixed_transfer_data_loader_old import data_processer
from util_tools import torch_ddpm_downscale_halo_hann

TRANS_DIR = "/project/def-mere/merra2/Ace_Transfer_Downscale/Transfer_model"
if TRANS_DIR not in sys.path:
    sys.path.append(TRANS_DIR)

from transfer_unet_fur_diffuser import SmallTransferUNet
torch.autograd.set_detect_anomaly(True)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=2000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="sample",
    clip_sample=False,
)


def get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, train_set, stride, AFG_only=False):
    start = time.time()

    if not os.path.exists(data_cache_path):
        os.makedirs(data_cache_path, exist_ok=True)

    file_path_g = "/project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc"
    file_path_m = "/project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc"
    file_path_ele = "/project/def-mere/merra2/g5nr/elevation/elevation_data.npy"

    print(f"Debug: {file_path_g}")

    if AFG_only:
        file_path_country = [
            "/project/def-mere/merra2/shapes/AFG_adm0.shp",
            "/project/def-mere/merra2/shapes/KGZ_adm0.shp",
        ]
    else:
        file_path_country = [
            "/project/def-mere/merra2/shapes/ARE_adm0.shp",
            "/project/def-mere/merra2/shapes/IRQ_adm0.shp",
            "/project/def-mere/merra2/shapes/KWT_adm0.shp",
            "/project/def-mere/merra2/shapes/QAT_adm0.shp",
            "/project/def-mere/merra2/shapes/SAU_adm0.shp",
            "/project/def-mere/merra2/shapes/DJI_adm0.shp",
        ]

    data_processor = data_processer()
    g_data, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data = data_processor.load_data(
        target_var,
        file_path_g,
        file_path_m,
        file_path_ele,
        file_path_country,
    )

    match_m_data = data_processor.unify_m_data(g_data[:10], m_data, G_lats, G_lons, M_lats, M_lons)
    match_m_data = match_m_data[1961:1961 + 763]

    print("m_data shape:", match_m_data.shape)
    print("g_data shape: ", g_data.shape)

    if g_data.shape[0] != 763 or match_m_data.shape[0] != 763:
        raise ValueError(
            f"Data Alignment Error: Required both G and M data to have 763 days for seasonal indexing, but found {g_data.shape[0]} for G5NR and {match_m_data.shape[0]} for MERRA2."
        )

    seasonal_indices = np.array(train_set) - 1

    seasonal_g_data = g_data[seasonal_indices]
    seasonal_m_data = match_m_data[seasonal_indices]
    days = train_set

    g_min, g_max = np.nanmin(seasonal_g_data), np.nanmax(seasonal_g_data)
    m_min, m_max = np.nanmin(seasonal_m_data), np.nanmax(seasonal_m_data)

    norm_params_path = os.path.join(data_cache_path, "norm_params.npz")
    np.savez(norm_params_path, g_min=g_min, g_max=g_max, m_min=m_min, m_max=m_max)
    print(f"Normalization parameters saved to {norm_params_path}")

    eps = 1e-9
    seasonal_g_data = (seasonal_g_data - g_min) / (g_max - g_min + eps)
    seasonal_m_data = (seasonal_m_data - m_min) / (m_max - m_min + eps)

    print("MERRA2 data shape for independently normalized seasonal training:", seasonal_m_data.shape)
    print("G5NR data shape for independently normalized seasonal training:", seasonal_g_data.shape)

    if (
        "X_high.npy" not in os.listdir(data_cache_path)
        or "X_low.npy" not in os.listdir(data_cache_path)
        or "X_ele.npy" not in os.listdir(data_cache_path)
        or "X_other.npy" not in os.listdir(data_cache_path)
        or "Y.npy" not in os.listdir(data_cache_path)
    ):
        X_high, X_low, X_ele, X_other, Y = data_processor.flatten(
            seasonal_g_data,
            seasonal_m_data,
            ele_data,
            [G_lats, G_lons],
            days,
            n_lag=n_lag,
            n_pred=n_pred,
            task_dim=task_dim,
            is_perm=True,
            return_Y=True,
            stride=stride,
        )

        np.save(os.path.join(data_cache_path, "X_high.npy"), X_high)
        np.save(os.path.join(data_cache_path, "X_low.npy"), X_low)
        np.save(os.path.join(data_cache_path, "X_ele.npy"), X_ele)
        np.save(os.path.join(data_cache_path, "X_other.npy"), X_other)
        np.save(os.path.join(data_cache_path, "Y.npy"), Y)
    else:
        print("Data is processed and saved, skipped data processing!")
    print("Data Processing Time: ", (time.time() - start) / 60, "mins")


def get_area_data(area):
    target_var = "DUEXTTAU"
    area = int(area)
    AFG_only = False if area != 0 else True

    file_path_g = "/project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc"
    file_path_m = "/project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc"
    file_path_ele = "/project/def-mere/merra2/g5nr/elevation/elevation_data.npy"
    if AFG_only:
        file_path_country = [
            "/project/def-mere/merra2/shapes/AFG_adm0.shp",
            "/project/def-mere/merra2/shapes/KGZ_adm0.shp",
        ]
    else:
        file_path_country = [
            "/project/def-mere/merra2/shapes/ARE_adm0.shp",
            "/project/def-mere/merra2/shapes/IRQ_adm0.shp",
            "/project/def-mere/merra2/shapes/KWT_adm0.shp",
            "/project/def-mere/merra2/shapes/QAT_adm0.shp",
            "/project/def-mere/merra2/shapes/SAU_adm0.shp",
            "/project/def-mere/merra2/shapes/DJI_adm0.shp",
        ]

    data_processor = data_processer()
    g_data, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data = data_processor.load_data(
        target_var,
        file_path_g,
        file_path_m,
        file_path_ele,
        file_path_country,
    )

    match_m_data_all = data_processor.unify_m_data(g_data[:10], m_data, G_lats, G_lons, M_lats, M_lons)
    match_m_data = match_m_data_all[1961:1961 + 763, :, :]

    if g_data.shape[0] != 763 or match_m_data.shape[0] != 763:
        raise ValueError(
            f"Data Alignment Error: Required 763 days for seasonal indexing, but found {g_data.shape[0]} for G5NR and {match_m_data.shape[0]} for MERRA2."
        )

    match_m_data_all = match_m_data_all[1961:1961 + 365 * 3, :, :]

    print("m_data shape:", match_m_data.shape)
    print("g_data shape: ", g_data.shape)
    days = list(range(1, 764))
    return g_data, match_m_data, ele_data, [G_lats, G_lons], days, match_m_data_all


class DownscaleDataset(Dataset):
    """Loads cached *.npy* arrays produced by *get_data()* in the pipeline.

        Each sample returns a dict with keys:
            * X_high  – (T_lag, 1, H, W) n_lags of G5NR past day data till t-1.
            * X_low   – (1, H, W):  current day MERRA2 at day t
            * X_ele   – (1, H, W)
            * X_other – (3,):    contains info of season day, lat and lon values.
            * y       – (H, W):  target G5NR at t.
    """

    def __init__(self, data_path):
        self.X_high = (
            torch.from_numpy(np.load(f"{data_path}/X_high.npy"))
            .float()
            .permute(0, 1, 4, 2, 3)
        )
        self.X_low = (
            torch.from_numpy(np.load(f"{data_path}/X_low.npy"))
            .float()
            .permute(0, 1, 4, 2, 3)
        )
        self.X_ele = (
            torch.from_numpy(np.load(f"{data_path}/X_ele.npy"))
            .float()
            .permute(0, 3, 1, 2)
        )

        raw_other = torch.from_numpy(np.load(f"{data_path}/X_other.npy")).float()
        self.day_scalar = raw_other[:, 2]
        self.X_other = raw_other[:, [0, 1, 3]]

        self.Y = torch.from_numpy(np.load(f"{data_path}/Y.npy")).float().squeeze(1)

        print("self.X_high shape:", self.X_high.shape)
        print("self.X_low shape:", self.X_low.shape)
        print("self.X_ele shape:", self.X_ele.shape)
        print("self.X_other shape:", self.X_other.shape)
        print("self.Y shape:", self.Y.shape)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return {
            "X_high": self.X_high[idx],
            "X_low": self.X_low[idx],
            "X_ele": self.X_ele[idx],
            "X_other": self.X_other[idx],
            "Y": self.Y[idx],
        }


class TorchCallbacks:
    """Mimics Keras-style ReduceLROnPlateau + EarlyStopping + Checkpoint for PyTorch only."""

    def __init__(self, optimizer, scheduler, patience_es, ckpt_path):
        self.opt = optimizer
        self.sched = scheduler
        self.patience_es = patience_es
        self.ckpt_path = ckpt_path
        self.best_val = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss, model):
        self.sched.step(val_loss)

        if val_loss < self.best_val:
            self.best_val = val_loss
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.ckpt_path)
            print("New best; checkpoint saved.")
        else:
            self.bad_epochs += 1

        stop = self.bad_epochs >= self.patience_es
        return stop


"""

The architecture of the small transfer model. Feel free to adjust the truncated layers with below information. 

>>> model_to_inspect = SmallTransferUNet(T_lag=40, H=16, W=16)
>>> print(model_to_inspect.unet)
UNet2DModel(
  (conv_in): Conv2d(40, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (time_proj): Timesteps()
  (time_embedding): TimestepEmbedding(
    (linear_1): Linear(in_features=64, out_features=256, bias=True)
    (act): SiLU()
    (linear_2): Linear(in_features=256, out_features=256, bias=True)
  )
  (down_blocks): ModuleList(
    (0): DownBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (1): DownBlock2D(
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 64, eps=1e-05, affine=True)
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (2): AttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Attention(
          (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
          (to_q): Linear(in_features=256, out_features=256, bias=True)
          (to_k): Linear(in_features=256, out_features=256, bias=True)
          (to_v): Linear(in_features=256, out_features=256, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): UpBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
          (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 384, eps=1e-05, affine=True)
          (conv1): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (1): AttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Attention(
          (group_norm): GroupNorm(32, 128, eps=1e-05, affine=True)
          (to_q): Linear(in_features=128, out_features=128, bias=True)
          (to_k): Linear(in_features=128, out_features=128, bias=True)
          (to_v): Linear(in_features=128, out_features=128, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=128, out_features=128, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 384, eps=1e-05, affine=True)
          (conv1): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
          (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 192, eps=1e-05, affine=True)
          (conv1): Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (2): UpBlock2D(
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 192, eps=1e-05, affine=True)
          (conv1): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
        )
        (1-2): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=256, out_features=64, bias=True)
          (norm2): GroupNorm(32, 64, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mid_block): UNetMidBlock2D(
    (attentions): ModuleList(
      (0): Attention(
        (group_norm): GroupNorm(32, 256, eps=1e-05, affine=True)
        (to_q): Linear(in_features=256, out_features=256, bias=True)
        (to_k): Linear(in_features=256, out_features=256, bias=True)
        (to_v): Linear(in_features=256, out_features=256, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (resnets): ModuleList(
      (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=256, out_features=256, bias=True)
        (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
  (conv_norm_out): GroupNorm(32, 64, eps=1e-05, affine=True)
  (conv_act): SiLU()
  (conv_out): Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
"""


class TinyEncoder(nn.Module):
    """
    SmallTransferUNet that returns the feature map from the deepest point of the U-Net, except output layer.
    """

    def __init__(self, ckpt: str, n_lag: int, H: int, W: int):
        super().__init__()
        core = SmallTransferUNet(n_lag, H, W).unet
        state = torch.load(ckpt, map_location="cpu")

        new_state_dict = {}
        for k, v in state.items():
            name = k.replace("unet.", "")
            new_state_dict[name] = v

        core.load_state_dict(new_state_dict)

        self.core = core.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        t = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        t_emb = self.core.time_proj(t)
        t_emb = self.core.time_embedding(t_emb)

        sample = self.core.conv_in(x)

        down_block_res_samples = (sample,)
        for downsample_block in self.core.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=t_emb, encoder_hidden_states=None
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=t_emb)
            down_block_res_samples += res_samples

        if self.core.mid_block is not None:
            sample = self.core.mid_block(sample, t_emb)

        return sample


class LargeUNet3D(nn.Module):
    """
    Output: 1×H×W G5NR_t
    Inputs
        • high_lags : (B, T, 1, H, W)   past 40 lags of G5NR tracing back from t.
        • low_t     : (B, 1, H, W)      MERRA-2 at t.
        • elev      : (B, 1, H, W)
        • misc      : (B, 3)            lat / lon / season scalar
    IMPORTANT Fix: add a final activation in the final outut layer to ensure non-negativity,
    and we also need to ensure pixel AOD values are between 0 and 1 (normalized.)
    """

    def __init__(self, tiny_ckpt: str, n_lag: int, H: int, W: int, c_other: int = 3):
        super().__init__()
        self.tiny = TinyEncoder(tiny_ckpt, n_lag, H, W)

        in_ch = 256 + 1 + 1 + c_other + 1
        self.unet3d = UNet3DConditionModel(
            sample_size=(H, W),
            in_channels=in_ch,
            out_channels=1,
            block_out_channels=(128, 256, 512),
            down_block_types=("DownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "UpBlock3D"),
        )

    def forward(self, x_t, high_lags, low_t, elev, misc_maps, t):
        """
        high_lags : (B, T, 1, H, W)  – from DataLoader, cuda tensor
        low_t     : (B, 1, H, W)
        elev      : (B, 1, H, W)
        misc_maps : (B, 3, H, W)
        """
        x = high_lags.squeeze(2)
        tiny_feat = self.tiny(x)

        target_size = low_t.shape[2:]
        tiny_feat = F.interpolate(tiny_feat, size=target_size, mode="bilinear", align_corners=False)
        if not torch.isfinite(tiny_feat).all():
            raise RuntimeError("tiny_feat exploded after F.interpolate")

        inp_4d = torch.cat(
            [
                x_t,
                tiny_feat,
                low_t,
                elev,
                misc_maps,
            ],
            dim=1,
        )
        inp_5d = inp_4d.unsqueeze(2)

        device_ = inp_4d.device

        encoder_hidden_states = torch.zeros(
            inp_4d.size(0),
            1,
            1024,
            dtype=torch.float,
            device=device_,
        )

        raw = self.unet3d(
            sample=inp_5d,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        output_bounded = torch.sigmoid(raw)

        return output_bounded.squeeze(2)


def train_large(data_dir: str, n_lag: int = 40, task_dim=(16, 16), epochs: int = 100, batch: int = 64, lr: float = 1e-4):
    H, W = task_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DownscaleDataset(data_dir)

    day_float = ds.day_scalar.numpy()
    day_id = np.round(day_float * 365).astype(int)
    uniq_days = np.unique(day_id)

    rng = np.random.default_rng(seed)
    val_days = rng.choice(uniq_days, size=int(0.20 * len(uniq_days)), replace=False)

    is_val = np.isin(day_id, val_days)
    train_ix = np.where(~is_val)[0]
    val_ix = np.where(is_val)[0]

    N_train, N_val = len(train_ix), len(val_ix)

    train_loader = DataLoader(
        ds,
        batch_size=batch,
        sampler=SubsetRandomSampler(train_ix, generator=torch.Generator().manual_seed(seed)),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds,
        batch_size=batch,
        sampler=SubsetRandomSampler(val_ix, generator=torch.Generator().manual_seed(seed)),
        num_workers=4,
        pin_memory=True,
    )

    head, tail = os.path.split(data_dir)
    season_dir, _ = os.path.split(head)
    season = os.path.basename(head)
    area = tail

    print("Loading small model checkpoint from the small transfer model.")
    tiny_ckpt = os.path.join(
        "/project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets",
        "small_transfer_model",
        "new_small_result",
        season,
        area,
        "transfer_unet_small.pt",
    )
    if not os.path.exists(tiny_ckpt):
        raise FileNotFoundError(f"Expected small-transfer checkpoint at {tiny_ckpt}")

    model = LargeUNet3D(tiny_ckpt, n_lag, H, W).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        patience=12,
        factor=0.1,
        min_lr=3e-6,
    )
    cb = TorchCallbacks(
        opt,
        sched,
        patience_es=24,
        ckpt_path=os.path.join(data_dir, "s2s_model.pt"),
    )
    use_amp = False
    scaler = torch.amp.GradScaler() if use_amp else None

    for ep in range(1, epochs + 1):
        model.train()
        trn = 0
        for batch_ in train_loader:
            y_clean = batch_["Y"].unsqueeze(1).to(device)

            bsz = y_clean.size(0)
            t = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(y_clean)
            x_t = noise_scheduler.add_noise(y_clean, noise, t)

            high = batch_["X_high"].to(device)
            low = batch_["X_low"][:, 0].to(device)
            ele = batch_["X_ele"].to(device)
            misc = (
                batch_["X_other"]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, H, W)
                .to(device)
            )

            if not torch.all(torch.isfinite(high)):
                print("NaN/inf found in X_high!")
            if not torch.all(torch.isfinite(low)):
                print("NaN/inf found in X_low!")
            if not torch.all(torch.isfinite(ele)):
                print("NaN/inf found in X_ele!")
            if not torch.all(torch.isfinite(misc)):
                print("NaN/inf found in extracted feature from transfer model!")
            if not torch.all(torch.isfinite(y_clean)):
                print("NaN/inf found in y_clean!")
            for tag, tensor in {
                "high": high,
                "low": low,
                "ele": ele,
                "misc": misc,
                "y": y_clean,
            }.items():
                if not torch.isfinite(tensor).all():
                    bad = tensor[~torch.isfinite(tensor)]
                    raise RuntimeError(
                        f"{tag} has {bad.numel()} non-finite values "
                        f"(min={bad.min()}, max={bad.max()})"
                    )

            pred_x0 = model(x_t, high, low, ele, misc, t)
            loss_val = F.mse_loss(pred_x0, y_clean)

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad(set_to_none=True)

            trn += loss_val.item() * bsz

        model.eval()
        val = 0
        with torch.no_grad():
            for batch_ in val_loader:
                y_clean = batch_["Y"].unsqueeze(1).to(device)

                bsz = y_clean.size(0)
                t = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noise = torch.randn_like(y_clean)
                x_t = noise_scheduler.add_noise(y_clean, noise, t)

                high = batch_["X_high"].to(device)
                low = batch_["X_low"][:, 0].to(device)
                ele = batch_["X_ele"].to(device)
                misc = (
                    batch_["X_other"]
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, H, W)
                    .to(device)
                )

                pred_x0 = model(x_t, high, low, ele, misc, t)
                val += F.mse_loss(pred_x0, y_clean).item() * bsz

        trn /= N_train
        val /= N_val
        print(f"Epoch {ep:02d}: train {trn:.4f} | val {val:.4f}")

        if cb.step(val, model):
            break

    print("Model is trained.")


if __name__ == "__main__":
    out_dir = sys.argv[1]
    head, tail = os.path.split(out_dir)
    area_id = int(tail[-1])

    train_set = np.load(os.path.join(head, "train_days.npy"))
    get_data(
        out_dir,
        "DUEXTTAU",
        40,
        1,
        TILE,
        train_set,
        stride=STRIDE_TRAIN,
        AFG_only=(area_id == 0),
    )

    check_path = os.path.join(out_dir, "s2s_model.pt")
    if not os.path.exists(check_path):
        print("We start training the model.")
        train_large(out_dir)
    else:
        print(f"Skip Training! The best model is already saved at {check_path}!")

    head, tail = os.path.split(out_dir)
    season = os.path.basename(head)
    area = tail

    SMALL_MODEL_BASE_PATH = "/project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets/small_transfer_model/new_small_result"
    tiny_ckpt_path = os.path.join(SMALL_MODEL_BASE_PATH, season, area, "transfer_unet_small.pt")
    large = LargeUNet3D(tiny_ckpt_path, 40, 16, 16).to("cpu")
    large.load_state_dict(torch.load(os.path.join(out_dir, "s2s_model.pt"), map_location="cpu"))
    ds = torch_ddpm_downscale_halo_hann.downscaler(large)
    print(
        f"[cfg] HALO={HALO} | USE_HANN={USE_HANN} | INFER_STEPS={INFER_STEPS} | TILE={TILE[0]}x{TILE[1]} | STRIDE_TRAIN={STRIDE_TRAIN} | STRIDE_INFER={STRIDE_INFER} | N_EST={N_EST}"
    )
    if STRIDE_INFER > (TILE[0] - 2 * HALO):
        raise ValueError(
            f"STRIDE_INFER={STRIDE_INFER} too large for TILE={TILE} with HALO={HALO} "
            f"(need ≤ {TILE[0] - 2 * HALO})."
        )

    g_data_raw, m_763_raw, ele, [G_lats, G_lons], days, _ = get_area_data(area_id)

    norm_params_path = os.path.join(out_dir, "norm_params.npz")
    params = np.load(norm_params_path)
    g_min, g_max = params["g_min"], params["g_max"]
    m_min, m_max = params["m_min"], params["m_max"]
    eps = 1e-9

    g_data = (g_data_raw - g_min) / (g_max - g_min + eps)
    m_763 = (m_763_raw - m_min) / (m_max - m_min + eps)

    test_set = np.load(os.path.join(head, "test_days.npy"))

    old_mean = os.path.join(out_dir, "downscaled_mean_strict.npy")
    old_days = os.path.join(out_dir, "pred_days_strict.npy")
    grid_mean = os.path.join(out_dir, "downscaled_mean_strict_grid.npy")
    grid_days = os.path.join(out_dir, "pred_days_strict_grid.npy")
    try:
        if os.path.exists(old_mean) and not os.path.exists(grid_mean):
            os.rename(old_mean, grid_mean)
            print(f"Renamed {old_mean} → {grid_mean}")
        if os.path.exists(old_days) and not os.path.exists(grid_days):
            os.rename(old_days, grid_days)
            print(f"Renamed {old_days} → {grid_days}")
    except Exception as e:
        print(f"[warn] rename step skipped due to: {e}")

    modes = ["strict", "lenient"]
    for mode in modes:
        out_mean = os.path.join(out_dir, f"downscaled_mean_{mode}.npy")
        out_days = os.path.join(out_dir, f"pred_days_{mode}.npy")

        if os.path.exists(out_mean) and os.path.exists(out_days):
            print(f"✓ {mode} outputs already exist – skipping.")
            continue

        hist_high = g_data.copy()
        preds, valid_days = [], []

        for t in test_set:
            if t <= 40:
                print(
                    "WARNING! This should not happen given that new seasonal split script is designed to prevent this. [skip] day {t} has <40 lags"
                )
                continue

            h_win = hist_high[t - 40 : t]
            l_win = m_763[t - 40 : t + 1]
            day_win = days[t - 40 : t + 1]

            a = ds.downscale(
                h_win,
                l_win,
                ele,
                [G_lats, G_lons, None, None],
                day_win,
                n_lag=40,
                n_pred=1,
                task_dim=TILE,
                n_est=N_EST,
                stride=STRIDE_INFER,
            )

            if a.size == 0:
                print(
                    f"[WARN] DANGEROUS!!! Downscaled data is all zeros! Empty downscale for day {t} – skipping."
                )
                continue
            pred_img = a[0]
            preds.append(pred_img)
            if mode == "strict":
                hist_high[t - 1] = pred_img
            valid_days.append(t)

        preds = np.asarray(preds, dtype=np.float32)
        np.save(out_mean, preds)
        np.save(out_days, np.asarray(valid_days, dtype=np.int32))
        print(f"▼ {mode}: saved {preds.shape} → {out_mean}")