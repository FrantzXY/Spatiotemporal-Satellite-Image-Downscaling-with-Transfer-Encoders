#### This is a script of small transfer model mapping temporal and spatial association of MERRA2 data only.
#### This small model is later used in diffusion based large downcale model in /project/def-mere/merra2/Ace_Transfer_Downscale/large_model/train_diffusion_with_transfer.py
#### Author: Jingwen Zhong, Date: 2025/07/04 


import numpy as np
import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import torch.nn.functional as F
import diffusers
from diffusers import UNet2DModel
import pandas as pd
import math
from torch.utils.data import Sampler

seed = 465
np.random.seed(seed)
torch.manual_seed(seed)

AHEAD = "/project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets"
if AHEAD not in sys.path:
    sys.path.append(AHEAD)
from util_tools.fixed_transfer_data_loader import data_processer
from util_tools import torch_downscale


def get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, train_set, stride, AFG_only=False):
    start = time.time()
    if not os.path.exists(data_cache_path):
        os.makedirs(data_cache_path, exist_ok=True)
    file_path_g = '/project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc'
    file_path_m = '/project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc'
    print(f"Debug: {file_path_g}")

    if AFG_only:
        file_path_country = ['/project/def-mere/merra2/shapes/AFG_adm0.shp',
                             '/project/def-mere/merra2/shapes/KGZ_adm0.shp']
    else:
        file_path_country = ['/project/def-mere/merra2/shapes/ARE_adm0.shp',
                             '/project/def-mere/merra2/shapes/IRQ_adm0.shp',
                             '/project/def-mere/merra2/shapes/KWT_adm0.shp',
                             '/project/def-mere/merra2/shapes/QAT_adm0.shp',
                             '/project/def-mere/merra2/shapes/SAU_adm0.shp',
                             '/project/def-mere/merra2/shapes/DJI_adm0.shp']

    data_processor = data_processer()
    g_data, m_data, [G_lats, G_lons, M_lats, M_lons], _ = data_processor.load_data(target_var,
                                                                                    file_path_g,
                                                                                    file_path_m,
                                                                                    file_path_ele=None,
                                                                                    file_path_country=file_path_country)

    match_m_data = m_data
    
    seasonal_indices = np.array(train_set) - 1
    seasonal_m_data = match_m_data[seasonal_indices]

    m_min, m_max = np.nanmin(seasonal_m_data), np.nanmax(seasonal_m_data)
    
    norm_params_path = os.path.join(data_cache_path, 'norm_params.npz')
    np.savez(norm_params_path, m_min=m_min, m_max=m_max)
    print(f"Normalization parameters saved to {norm_params_path}")

    eps = 1e-9
    seasonal_m_data = (seasonal_m_data - m_min) / (m_max - m_min + eps)
    print('m_data shape for seasonal training:', seasonal_m_data.shape)
    seasonal_m_data = seasonal_m_data.filled(0)

    np.save(os.path.join(data_cache_path, 'seasonal_m_data.npy'), seasonal_m_data.astype('float32'))
    print('Seasonal data cube saved, skipping patch generation.')
    
    print('Data Processing Time: ', (time.time()-start)/60, 'mins')


def get_area_data(area_id: str, target_var='DUEXTTAU'):
    """Load *only* MERRA-2 cube and its lat/lon for the requested region."""
    area = int(area_id)
    AFG_only = area == 0

    file_path_g = '/project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc'
    file_path_m = '/project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc'

    file_path_country = (
        ['/project/def-mere/merra2/shapes/AFG_adm0.shp',
         '/project/def-mere/merra2/shapes/KGZ_adm0.shp']
        if AFG_only else
        ['/project/def-mere/merra2/shapes/ARE_adm0.shp',
         '/project/def-mere/merra2/shapes/IRQ_adm0.shp',
         '/project/def-mere/merra2/shapes/KWT_adm0.shp',
         '/project/def-mere/merra2/shapes/QAT_adm0.shp',
         '/project/def-mere/merra2/shapes/SAU_adm0.shp',
         '/project/def-mere/merra2/shapes/DJI_adm0.shp']
    )

    proc = data_processer()
    g_data, m_data, [G_lats, G_lons, M_lats, M_lons], _ = proc.load_data(target_var, file_path_g, file_path_m,file_path_ele=None,file_path_country=file_path_country)
    match_m_data = m_data
    return match_m_data, M_lats, M_lons



class DayBatchSampler(Sampler):
    """
    Yields batches of patch indices grouped by 'day', without materializing
    a giant day_per_patch array. 100% deterministic with the given seed.
    """
    def __init__(self, n_valid_days:int, patches_per_day:int,
                 batch_size:int, val_frac:float=0.20, seed:int=465, train:bool=True):
        self.n_valid_days   = n_valid_days
        self.ppd            = patches_per_day
        self.batch_size     = batch_size
        self.rng            = np.random.default_rng(seed)
        all_days  = np.arange(n_valid_days)
        n_val     = max(1, int(val_frac * len(all_days)))
        val_days  = set(self.rng.choice(all_days, size=n_val, replace=False))
        self.days = [d for d in all_days if ((d not in val_days) if train else (d in val_days))]
        if train:
            self.rng.shuffle(self.days)

    def __iter__(self):
        buf = []
        for d in self.days:
            start = d * self.ppd
            day_idxs = np.arange(start, start + self.ppd)
            if self.batch_size >= self.ppd:
                pass
            self.rng.shuffle(day_idxs)
            for idx in day_idxs:
                buf.append(int(idx))
                if len(buf) == self.batch_size:
                    yield buf
                    buf = []
        if buf:
            yield buf

    def __len__(self):
        total = len(self.days) * self.ppd
        return math.ceil(total / self.batch_size)


class AerosolDataset(Dataset):
    def __init__(self, data_path, n_lag=40, n_pred=1, task_dim=(16, 16), stride=8):
        self.n_lag = n_lag
        self.n_pred = n_pred
        self.task_h, self.task_w = task_dim
        self.stride = stride

        self.data = np.load(os.path.join(data_path, 'seasonal_m_data.npy'), mmap_mode='r')
        
        n_days, self.full_h, self.full_w = self.data.shape
        
        self.rows = (self.full_h - self.task_h) // self.stride + 1
        self.cols = (self.full_w - self.task_w) // self.stride + 1
        self.patches_per_day = self.rows * self.cols
        
        self.n_valid_days = n_days - self.n_lag - self.n_pred + 1
        self.total_patches = self.n_valid_days * self.patches_per_day

        print(f"Dataset initialized. Total valid patches: {self.total_patches}")

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        day_idx_offset = idx // self.patches_per_day
        patch_in_day_idx = idx % self.patches_per_day
        
        row_idx = (patch_in_day_idx // self.cols) * self.stride
        col_idx = (patch_in_day_idx % self.cols) * self.stride
        
        t = day_idx_offset + self.n_lag - 1
        
        x_slice = self.data[t - (self.n_lag - 1) : t + 1, row_idx : row_idx + self.task_h, col_idx : col_idx + self.task_w]
        y_slice = self.data[t + 1 : t + 1 + self.n_pred, row_idx : row_idx + self.task_h, col_idx : col_idx + self.task_w]

        x = torch.from_numpy(x_slice.copy()).float()
        y = torch.from_numpy(y_slice.copy()).float().squeeze(0)
        
        return x, y  


class SmallTransferUNet(nn.Module):
    def __init__(self, T_lag:int, H:int, W:int, blocks=(64, 128, 256)) -> None:
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=(H, W),
            in_channels=T_lag,
            out_channels=1,
            block_out_channels=blocks,
            down_block_types=(
                "DownBlock2D","DownBlock2D",
                "AttnDownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D","AttnUpBlock2D",
                "UpBlock2D"
            ),
        )

    def forward(self, x):
        t = torch.zeros(x.size(0), dtype=torch.long, device=x.device) 
        return self.unet(x, t).sample


class TorchCallbacks:
    """Mimics Keras-style ReduceLROnPlateau + EarlyStopping + Checkpoint for PyTorch only."""
    def __init__(self, optimizer, scheduler, patience_es, ckpt_path):
        self.opt         = optimizer
        self.sched       = scheduler
        self.patience_es = patience_es
        self.ckpt_path   = ckpt_path
        self.best_val    = float("inf")
        self.bad_epochs  = 0

    def step(self, val_loss, model):
        self.sched.step(val_loss)

        if val_loss < self.best_val:
            self.best_val   = val_loss
            self.bad_epochs = 0
            torch.save(model.state_dict(), self.ckpt_path)
            print("We achieved new best; checkpoint saved.")
        else:
            self.bad_epochs += 1

        stop = self.bad_epochs >= self.patience_es
        return stop



def main(data_cache_path:str,
         n_lag:int=40,
         task_dim=(16,16),
         epochs:int=20,
         batch_size:int=32,
         lr:float=5e-4, stride = 8):

    H, W = task_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ds = AerosolDataset(data_cache_path)

    n_valid_days     = ds.n_valid_days
    patches_per_day  = ds.patches_per_day

    train_batch_sampler = DayBatchSampler(n_valid_days, patches_per_day,
                                        batch_size=batch_size, val_frac=0.20,
                                        seed=seed, train=True)
    val_batch_sampler   = DayBatchSampler(n_valid_days, patches_per_day,
                                        batch_size=batch_size, val_frac=0.20,
                                        seed=seed, train=False)

    train_loader = DataLoader(ds, batch_sampler=train_batch_sampler,
                            num_workers=2, prefetch_factor=2,
                            pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(ds, batch_sampler=val_batch_sampler,
                            num_workers=2, prefetch_factor=2,
                            pin_memory=False, persistent_workers=False)


    model = SmallTransferUNet(n_lag, H, W).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=3, min_lr=1e-6)
    cb = TorchCallbacks(opt, sched, patience_es=6, ckpt_path=os.path.join(data_cache_path, "transfer_unet_small.pt")) 
    loss_fn = nn.L1Loss()   


    best_val = float("inf")
    for epoch in range(1, epochs+1):
        model.train(); trn_loss = 0.0; n_trn_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.unsqueeze(1).to(device)   
            pred   = model(xb)
            loss   = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = xb.size(0)
            trn_loss  += loss.item() * bs
            n_trn_seen += bs

        model.eval(); val_loss = 0.0; n_val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.unsqueeze(1).to(device)
                bs = xb.size(0)
                val_loss  += loss_fn(model(xb), yb).item() * bs
                n_val_seen += bs

        trn_loss /= max(1, n_trn_seen); val_loss /= max(1, n_val_seen)
        print(f"Epoch {epoch:02d}: train MAE {trn_loss:.4f}  |  val MAE {val_loss:.4f}")

        if cb.step(val_loss, model):
            print("Early-stopping – training halted."); break

    print("Our training is finished.")



if __name__ == '__main__':
    start = time.time()
    data_cache_path = sys.argv[1]
    head, tail = os.path.split(data_cache_path)
    area = tail[-1]
    AFG_only = True if int(area) == 0 else False
    train_set = np.load(os.path.join(head, 'train_days.npy'))

    n_lag = 40
    n_pred = 1
    stride = 8
    task_dim = [16, 16]
    target_var = 'DUEXTTAU'
    test_ratio = 0.1
    epochs = 20
    latent_space_dim = 10
    n_est = 5

    get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, train_set, AFG_only=AFG_only,
             stride=stride)
    
    check_path = os.path.join(data_cache_path, "transfer_unet_small.pt")
    if not os.path.exists(check_path):
        print(f"We start training the model.")
        main(data_cache_path, batch_size=32)
    else:
        print(f"Skip Training! The best model is already saved at {check_path}!")
        

    print("Down-scaling hold-out test days for MERRA2 small transfer model…")
    
    l_data_full_raw, M_lats, M_lons = get_area_data(area)

    norm_params_path = os.path.join(data_cache_path, 'norm_params.npz')
    params = np.load(norm_params_path)
    m_min, m_max = params['m_min'], params['m_max']
    eps = 1e-9

    l_data_full = (l_data_full_raw - m_min) / (m_max - m_min + eps)

    test_set = np.load(os.path.join(head, "test_days.npy"))

    print(f"\n[DEBUG] Full data shape for Area {area} is: {l_data_full.shape}")
    print(f"[DEBUG] Test set contains {len(test_set)} days. Max: {np.max(test_set)}, Min: {np.min(test_set)}")

    preds_out = []

    class _SqueezeWrapper(nn.Module):
        def __init__(self, base):
            super().__init__(); self.base = base
        def forward(self, x, *args, **kw):
            if x.dim() == 5 and x.size(-1) == 1:
                x = x.squeeze(-1)
            return self.base(x, *args, **kw)

    small_net = SmallTransferUNet(n_lag, *task_dim)
    small_net.load_state_dict(torch.load(
            os.path.join(data_cache_path, "transfer_unet_small.pt"),
            map_location="cpu"))
    small_net.to("cuda").eval()

    wrapped_net = _SqueezeWrapper(small_net)


    from util_tools.torch_downscale import downscaler as TorchDown
    ds = TorchDown(wrapped_net) 

    for t_day in test_set:

        if t_day <= n_lag:
            print(f"[INFO] Skipping test day {t_day} because it is less than or equal to the n_lag of {n_lag}.")
            continue
        t0 = t_day - 1
        l_win = l_data_full[t0 - n_lag : t0 + 1]
        print(f"[DEBUG] Day {t_day}: Slicing l_data_full[{t0 - n_lag}:{t0 + 1}]. Resulting l_win shape: {l_win.shape}")

        y_hat = ds.transfer_downscale(
            l_win, M_lats, M_lons,      
            n_lag=n_lag, n_pred=1,
            task_dim=task_dim, stride=stride, n_est=n_est
        )
        preds_out.append(y_hat[0])

    preds_out = np.asarray(preds_out, dtype=np.float32)
    np.save(os.path.join(data_cache_path, "downscaled_mean.npy"), preds_out)
    print("Wrote", preds_out.shape, "to downscaled_mean.npy")
