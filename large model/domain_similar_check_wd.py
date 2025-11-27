# A script for domain similarity check to justify the usage of transfer learning betwen MERRA2 and G5NR domains.
# Author: Yang Xiang, Date: 2025-06-20
# Wasserstein-1D domain similarity between MERRA-2 and G5NR
# Papers: https://arxiv.org/abs/1707.01217       https://arxiv.org/abs/2009.00909

"""
Results: WD close to zero, high domain similarity, good justification for TL.
Variable: DUEXTTAU | WD on [0,1] after log+joint min-max. Streaming, float32.

[Area0] box=(29.223,43.406)×(60.308,80.427)
  Per-day WD (overlap 763d): mean=0.0282, median=0.0237, p10=0.0114, p90=0.0478, n_days=763
  Pooled WD (ALL MERRA vs G5NR): 0.0080

[Area1] box=(10.650,37.643)×(34.275,56.601)
  Per-day WD (overlap 763d): mean=0.0237, median=0.0210, p10=0.0087, p90=0.0371, n_days=763
  Pooled WD (ALL MERRA vs G5NR): 0.0151
"""

import numpy as np
from netCDF4 import Dataset
import geopandas as gpd
import pandas as pd
from scipy.stats import wasserstein_distance

VAR   = "DUEXTTAU"
EPS   = 1e-7
DTYPE = np.float32

PATH_G = "/project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc"
PATH_M = "/project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc"

G5NR_N_DAYS = 763
MERRA2_G5NR_START_OFFSET = 1961

AFG_KGZ = [
    "/project/def-mere/merra2/shapes/AFG_adm0.shp",
    "/project/def-mere/merra2/shapes/KGZ_adm0.shp",
]
ARAB_CLUSTER = [
    "/project/def-mere/merra2/shapes/ARE_adm0.shp",
    "/project/def-mere/merra2/shapes/IRQ_adm0.shp",
    "/project/def-mere/merra2/shapes/KWT_adm0.shp",
    "/project/def-mere/merra2/shapes/QAT_adm0.shp",
    "/project/def-mere/merra2/shapes/SAU_adm0.shp",
    "/project/def-mere/merra2/shapes/DJI_adm0.shp",
]


CHUNK_DAYS = 32
BINS       = 256
SAMPLE_EVERY_N_DAYS = 1  

def box_from_shapes(paths, buffer_frac=0.01):
    gdf = gpd.read_file(paths[0])
    for p in paths[1:]:
        gdf = pd.concat([gdf, gpd.read_file(p)], ignore_index=True)
    lon_min, lat_min, lon_max, lat_max = gdf.total_bounds
    dlat = (lat_max - lat_min) * buffer_frac
    dlon = (lon_max - lon_min) * buffer_frac
    return dict(
        lat_min=float(lat_min - dlat),
        lat_max=float(lat_max + dlat),
        lon_min=float(lon_min - dlon),
        lon_max=float(lon_max + dlon),
    )

REGIONS = {
    "Area0": box_from_shapes(AFG_KGZ),
    "Area1": box_from_shapes(ARAB_CLUSTER),
}

def _as_plain(x, dtype=DTYPE):
    if isinstance(x, np.ma.MaskedArray):
        x = x.filled(np.nan)
    return np.array(x, dtype=dtype, copy=False)

def _slice_box(lat_arr, lon_arr, box):
    i_lat = np.where((lat_arr >= box["lat_min"]) & (lat_arr <= box["lat_max"]))[0]
    i_lon = np.where((lon_arr >= box["lon_min"]) & (lon_arr <= box["lon_max"]))[0]
    if i_lat.size == 0 or i_lon.size == 0:
        raise ValueError(f"Lat/lon box selects no grid cells: {box}")
    return i_lat, i_lon

def _contiguous_bounds(idx):
    return int(idx[0]), int(idx[-1]) + 1

def _precompute_regrid_indices(m_lats, m_lons, g_lats_sub, g_lons_sub):
    lat_idx = np.argmin(np.abs(m_lats[None, :] - g_lats_sub[:, None]), axis=1).astype(np.int64)
    lon_idx = np.argmin(np.abs(m_lons[None, :] - g_lons_sub[:, None]), axis=1).astype(np.int64)
    return lat_idx, lon_idx

def _log_minmax_update(block, mn, mx, eps=EPS):
    b = np.log10(np.maximum(block, eps)).astype(DTYPE, copy=False)
    if np.isfinite(b).any():
        mn = min(mn, float(np.nanmin(b)))
        mx = max(mx, float(np.nanmax(b)))
    return mn, mx

def _normalize_inplace(block, mn, mx, eps=EPS):
    b = np.log10(np.maximum(block, eps)).astype(DTYPE, copy=False)
    if mx <= mn or not np.isfinite(mn) or not np.isfinite(mx):
        b[:] = 0
    else:
        b = (b - mn) / (mx - mn)
        np.clip(b, 0, 1, out=b)
    return b

def _hist_accumulate(norm_block, hist, bins=BINS):
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=DTYPE)
    x = norm_block[np.isfinite(norm_block)]
    if x.size:
        hist += np.histogram(np.clip(x, 0, 1), edges)[0]
    return hist

def joint_minmax_overlap(dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1, m_lat_idx, m_lon_idx,
                         m_t0, m_t1, chunk=CHUNK_DAYS):
    mn, mx = np.inf, -np.inf
    T = int(gvar.shape[0])
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        g_blk = _as_plain(gvar[s:e, g_lat0:g_lat1, g_lon0:g_lon1])
        mn, mx = _log_minmax_update(g_blk, mn, mx)
    for s in range(m_t0, m_t1, chunk):
        e = min(s + chunk, m_t1)
        m_full = _as_plain(mvar[s:e, :, :])
        m_sel  = m_full[:, m_lat_idx[:, None], m_lon_idx[None, :]]
        mn, mx = _log_minmax_update(m_sel, mn, mx)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        mn, mx = 0.0, 1.0
    return mn, mx

def joint_minmax_full(dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1, m_lat_idx, m_lon_idx,
                      chunk=CHUNK_DAYS):
    mn, mx = np.inf, -np.inf
    Tg = int(gvar.shape[0])
    for s in range(0, Tg, chunk):
        e = min(s + chunk, Tg)
        g_blk = _as_plain(gvar[s:e, g_lat0:g_lat1, g_lon0:g_lon1])
        mn, mx = _log_minmax_update(g_blk, mn, mx)
    Tm = int(mvar.shape[0])
    for s in range(0, Tm, chunk):
        e = min(s + chunk, Tm)
        m_full = _as_plain(mvar[s:e, :, :])
        m_sel  = m_full[:, m_lat_idx[:, None], m_lon_idx[None, :]]
        mn, mx = _log_minmax_update(m_sel, mn, mx)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        mn, mx = 0.0, 1.0
    return mn, mx

def per_day_wd_overlap(dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1,
                       m_lat_idx, m_lon_idx, m_t0, m_t1,
                       mn, mx, step=SAMPLE_EVERY_N_DAYS, chunk=CHUNK_DAYS):
    wds = []
    Tg = int(gvar.shape[0])
    for t in range(0, Tg, step):
        g_blk = _as_plain(gvar[t:t+1, g_lat0:g_lat1, g_lon0:g_lon1])[0]
        g_nrm = _normalize_inplace(g_blk, mn, mx)
        tt = m_t0 + t
        m_full = _as_plain(mvar[tt:tt+1, :, :])[0]
        m_sel  = m_full[m_lat_idx[:, None], m_lon_idx[None, :]]
        m_nrm  = _normalize_inplace(m_sel, mn, mx)
        g_flat = g_nrm.ravel()
        m_flat = m_nrm.ravel()
        valid  = np.isfinite(g_flat) & np.isfinite(m_flat)
        if valid.any():
            wds.append(wasserstein_distance(g_flat[valid], m_flat[valid]))
    return np.asarray(wds, dtype=np.float32)

def pooled_wd_full_hist(dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1,
                        m_lat_idx, m_lon_idx, mn, mx, bins=BINS, chunk=CHUNK_DAYS):
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=DTYPE)
    hg = np.zeros(bins, dtype=np.float64)
    hm = np.zeros(bins, dtype=np.float64)
    Tg = int(gvar.shape[0])
    for s in range(0, Tg, chunk):
        e = min(s + chunk, Tg)
        g_blk = _as_plain(gvar[s:e, g_lat0:g_lat1, g_lon0:g_lon1])
        g_nrm = _normalize_inplace(g_blk, mn, mx)
        hg += np.histogram(np.clip(g_nrm[np.isfinite(g_nrm)], 0, 1), edges)[0]
    Tm = int(mvar.shape[0])
    for s in range(0, Tm, chunk):
        e = min(s + chunk, Tm)
        m_full = _as_plain(mvar[s:e, :, :])
        m_sel  = m_full[:, m_lat_idx[:, None], m_lon_idx[None, :]]
        m_nrm  = _normalize_inplace(m_sel, mn, mx)
        hm += np.histogram(np.clip(m_nrm[np.isfinite(m_nrm)], 0, 1), edges)[0]
    if hg.sum() == 0 or hm.sum() == 0:
        return np.nan
    cg = np.cumsum(hg) / hg.sum()
    cm = np.cumsum(hm) / hm.sum()
    return float(np.sum(np.abs(cg - cm)) * (1.0 / bins))

def main():
    print(f"\nVariable: {VAR} | WD on [0,1] after log+joint min-max. Streaming, float32.\n")
    with Dataset(PATH_G) as dg, Dataset(PATH_M) as dm:
        gvar = dg.variables[VAR]
        g_lats = _as_plain(dg.variables["lat"][:])
        g_lons = _as_plain(dg.variables["lon"][:])

        mvar = dm.variables[VAR]
        m_lats = _as_plain(dm.variables["lat"][:])
        m_lons = _as_plain(dm.variables["lon"][:])

        for name, box in REGIONS.items():
            i_lat, i_lon = _slice_box(g_lats, g_lons, box)
            g_lat0, g_lat1 = _contiguous_bounds(i_lat)
            g_lon0, g_lon1 = _contiguous_bounds(i_lon)
            g_lats_sub = g_lats[g_lat0:g_lat1]
            g_lons_sub = g_lons[g_lon0:g_lon1]

            m_lat_idx, m_lon_idx = _precompute_regrid_indices(m_lats, m_lons, g_lats_sub, g_lons_sub)
            m_t0 = MERRA2_G5NR_START_OFFSET
            m_t1 = m_t0 + min(G5NR_N_DAYS, int(gvar.shape[0]))
            mn_ov, mx_ov = joint_minmax_overlap(
                dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1,
                m_lat_idx, m_lon_idx, m_t0, m_t1
            )
            wds = per_day_wd_overlap(
                dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1,
                m_lat_idx, m_lon_idx, m_t0, m_t1, mn_ov, mx_ov,
                step=SAMPLE_EVERY_N_DAYS
            )
            if wds.size:
                stats = dict(mean=float(np.mean(wds)),
                             median=float(np.median(wds)),
                             p10=float(np.percentile(wds,10)),
                             p90=float(np.percentile(wds,90)),
                             n_days=int(wds.size))
            else:
                stats = dict(mean=np.nan, median=np.nan, p10=np.nan, p90=np.nan, n_days=0)
            mn_full, mx_full = joint_minmax_full(
                dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1,
                m_lat_idx, m_lon_idx
            )
            pooled_full = pooled_wd_full_hist(
                dg, gvar, dm, mvar, g_lat0, g_lat1, g_lon0, g_lon1,
                m_lat_idx, m_lon_idx, mn_full, mx_full
            )

            print(f"[{name}] box=({box['lat_min']:.3f},{box['lat_max']:.3f})×({box['lon_min']:.3f},{box['lon_max']:.3f})")
            print(f"  Per-day WD (overlap 763d): mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
                  f"p10={stats['p10']:.4f}, p90={stats['p90']:.4f}, n_days={stats['n_days']}")
            print(f"  Pooled WD (ALL MERRA vs G5NR): {pooled_full:.4f}\n")

if __name__ == "__main__":
    main()
