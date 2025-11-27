###_______ Author: Yige Yan_______###

from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import geopandas as gpd
except Exception:
    gpd = None
try:
    from shapely.geometry import Point
    from shapely.prepared import prep as shapely_prep
    try:
        from shapely import vectorized as shp_vec
    except Exception:
        shp_vec = None
except Exception:
    Point = None
    shapely_prep = None
    shp_vec = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semivariogram + ACF/PACF for G5NR/MERRA-2/Pred per Season/Area.")
    p.add_argument("mode", choices=["native", "id", "id-native"],
                   help="native: G5NR vs MERRA-2; id: Pred vs G5NR; id-native: MERRA-2 vs G5NR on ID dates.")
    p.add_argument("--rundir", required=True, type=str,
                   help="Root with Season*/Area*/ (e.g., new_large_ddpm_3_with_transfer_result).")
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--area", required=True, type=int)
    p.add_argument("--outdir", required=True, type=str)

    p.add_argument("--dates", nargs="*", default=None,
                   help="Explicit YYYY-MM-DD dates to plot semivariograms for.")
    p.add_argument("--n-days", type=int, default=3)
    p.add_argument("--sample", choices=["even", "random"], default="even")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-km", type=float, default=600.0)
    p.add_argument("--nbins", type=int, default=24)
    p.add_argument("--pairs", type=int, default=30000)
    p.add_argument("--mask", action="store_true", help="Restrict to Area polygons if shapefiles available.")
    p.add_argument("--semivar-all", action="store_true",
                   help="Also plot an ALL-days aggregated semivariogram for each field.")
    p.add_argument("--sync-y", choices=["off", "pair", "all"], default="pair",
                   help="Y-axis sync: off, per-day pair, or across all figures.")

    p.add_argument("--acf-pacf", action="store_true", help="Also compute ACF/PACF of region-mean series.")
    p.add_argument("--acf-lags", type=int, default=30)

    p.add_argument("--shapes-root", type=str, default="/project/def-mere/merra2/shapes")
    p.add_argument("--g5nr-nc", type=str,
                   default="/project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc")
    p.add_argument("--merra2-nc", type=str,
                   default="/project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc")
    p.add_argument("--elev-npy", type=str,
                   default="/project/def-mere/merra2/g5nr/elevation/elevation_data.npy")
    return p.parse_args(argv)


def _attach_project_to_syspath(rundir: Path):
    base = rundir.parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))


REGION_SHAPES_AFG = ["AFG_adm0.shp", "KGZ_adm0.shp"]
REGION_SHAPES_GULF = ["ARE_adm0.shp", "IRQ_adm0.shp", "KWT_adm0.shp", "QAT_adm0.shp", "SAU_adm0.shp", "DJI_adm0.shp"]


def _load_polygons(area_id: int, shapes_root: str):
    if gpd is None:
        return None
    names = REGION_SHAPES_AFG if int(area_id) == 0 else REGION_SHAPES_GULF
    gdfs = []
    for nm in names:
        p = Path(shapes_root) / nm
        if p.exists():
            try:
                gdfs.append(gpd.read_file(p))
            except Exception:
                pass
    if not gdfs:
        return None
    import pandas as _pd
    gdf = gpd.GeoDataFrame(_pd.concat(gdfs, ignore_index=True))
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass
    return gdf


def _region_mask(area_id: int, lats: np.ndarray, lons: np.ndarray, shapes_root: str):
    gdf = _load_polygons(area_id, shapes_root)
    if gdf is None or gdf.empty:
        return None
    try:
        geom = gdf.union_all()
    except Exception:
        geom = gdf.unary_union
    LON, LAT = np.meshgrid(lons, lats)
    if shp_vec is not None:
        try:
            return shp_vec.contains(geom, LON, LAT)
        except Exception:
            pass
    if Point is None or shapely_prep is None:
        return None
    prep = shapely_prep(geom)
    H, W = LON.shape
    out = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            out[i, j] = prep.contains(Point(float(LON[i, j]), float(LAT[i, j])))
    return out


def _days_to_dates_strict(days_arr) -> list[str]:
    base = pd.Timestamp("2005-05-15")
    days = np.asarray(days_arr).astype(int)
    return [str((base + pd.Timedelta(days=int(d) - 1)).date()) for d in days]


def _pick_indices(n: int, k: int, mode: str, seed: int) -> np.ndarray:
    k = max(1, min(k, n))
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=k, replace=False))
        return idx.astype(int)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def _mask_apply(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return x
    arr = x.copy()
    arr[~mask] = np.nan
    return arr


def _nanmean2d(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def empirical_variogram(field: np.ndarray,
                        lats: np.ndarray, lons: np.ndarray,
                        mask: np.ndarray | None,
                        max_km: float, nbins: int,
                        n_pairs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    H, W = field.shape
    LON, LAT = np.meshgrid(lons, lats)
    vals = field.reshape(-1)
    latv = LAT.reshape(-1)
    lonv = LON.reshape(-1)
    if mask is not None:
        mv = mask.reshape(-1)
        ok = mv & np.isfinite(vals)
    else:
        ok = np.isfinite(vals)
    idx = np.where(ok)[0]
    bins = np.linspace(0, max_km, nbins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    if idx.size < 2:
        return centers, np.full(nbins, np.nan), np.zeros(nbins, int)

    i1 = rng.choice(idx, size=n_pairs)
    i2 = rng.choice(idx, size=n_pairs)
    same = (i1 == i2)
    if np.any(same):
        keep = ~same
        i1 = i1[keep]
        i2 = i2[keep]

    v1 = vals[i1]
    v2 = vals[i2]
    d2 = (v1 - v2) ** 2
    dist = _haversine_km(latv[i1], lonv[i1], latv[i2], lonv[i2])

    which = np.digitize(dist, bins) - 1
    gamma = np.full(nbins, np.nan, dtype=float)
    counts = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        sel = which == b
        counts[b] = int(np.count_nonzero(sel))
        if counts[b] > 0:
            gamma[b] = 0.5 * float(np.nanmean(d2[sel]))
    return centers, gamma, counts


def aggregate_variogram_over_days(F: np.ndarray,
                                  lats: np.ndarray, lons: np.ndarray,
                                  mask: np.ndarray | None,
                                  max_km: float, nbins: int,
                                  pairs_per_day: int, seed: int = 0):
    sum_w = np.zeros(nbins, float)
    sum_n = np.zeros(nbins, int)
    centers = None
    T = F.shape[0]
    for t in range(T):
        field_t = _mask_apply(F[t], mask)
        c, g, n = empirical_variogram(field_t, lats, lons, mask,
                                      max_km, nbins, pairs_per_day,
                                      seed=(hash(("agg", int(seed), t)) % 2 ** 31))
        if centers is None:
            centers = c
        ok = np.isfinite(g) & (n > 0)
        sum_w[ok] += g[ok] * n[ok]
        sum_n[ok] += n[ok]
    gamma = np.full(nbins, np.nan, float)
    nz = sum_n > 0
    gamma[nz] = sum_w[nz] / sum_n[nz]
    return centers, gamma, sum_n


def acf_1d(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return np.array([np.nan], float)
    x = x - np.nanmean(x)
    if np.isnan(x).any():
        t = np.arange(n)
        good = ~np.isnan(x)
        x = x.copy()
        x[~good] = np.interp(t[~good], t[good], x[good])
    nlags_eff = max(0, min(int(nlags), n - 1))
    r_full = np.correlate(x, x, mode='full')
    start = n - 1
    stop = start + nlags_eff + 1
    r = r_full[start:stop]
    r /= r[0] if r[0] != 0 else 1.0
    return r


def pacf_yw(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return np.array([np.nan], float)
    nlags_eff = max(0, min(int(nlags), n - 1))
    if nlags_eff == 0:
        return np.array([1.0], float)
    r = acf_1d(x, nlags_eff)
    pacf = np.zeros(nlags_eff + 1, float)
    pacf[0] = 1.0
    phi = np.zeros((nlags_eff + 1, nlags_eff + 1), float)
    sig = np.zeros(nlags_eff + 1, float)
    sig[0] = r[0]
    for k in range(1, nlags_eff + 1):
        num = r[k] - np.sum(phi[1:k, k - 1] * r[1:k][::-1])
        den = sig[k - 1] if sig[k - 1] != 0 else 1.0
        phi[k, k] = num / den
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)
        pacf[k] = phi[k, k]
    return pacf


def _spherical_model(h, nugget, sill, a):
    h = np.asarray(h, float)
    out = np.full_like(h, nugget + sill, dtype=float)
    t = np.clip(h / max(a, 1e-9), 0, 1)
    inside = h <= a
    out[inside] = nugget + sill * (1.5 * t[inside] - 0.5 * t[inside] ** 3)
    return out


def _fit_spherical_quick(h, g, counts):
    h = np.asarray(h, float)
    g = np.asarray(g, float)
    w = np.asarray(counts, float)
    ok = np.isfinite(g) & (w > 0)
    if ok.sum() < 3:
        nug = max(0.0, np.nanmin(g))
        sill = max(1e-9, np.nanmax(g) - nug)
        a = h[ok][-1] if ok.any() else (h[-1] if h.size else 1.0)
        return nug, sill, a
    h_ok, g_ok, w_ok = h[ok], g[ok], w[ok]
    nug = max(0.0, float(np.nanmedian(g_ok[:min(3, g_ok.size)])))
    sill = max(1e-9, float(np.nanmedian(g_ok[-max(3, int(0.2 * g_ok.size)):])) - nug)
    target = nug + 0.95 * sill
    try:
        a = float(h_ok[np.where(g_ok >= target)[0][0]])
    except Exception:
        a = float(h_ok[-1])
    m = _spherical_model(h_ok, nug, sill, a)
    adj = np.sum(w_ok * (g_ok - m)) / (np.sum(w_ok) + 1e-9)
    sill = max(1e-9, sill + adj)
    return nug, sill, a


def _plot_variogram_paper(centers_km, gamma_vals, counts, title, outpng,
                          normalize_lag=True, ylims: Tuple[float, float] | None = None):
    centers_km = np.asarray(centers_km, float)
    gamma_vals = np.asarray(gamma_vals, float)
    counts = np.asarray(counts, int)

    x = centers_km
    xlabel = "Lag (ℓ)"
    if normalize_lag and np.isfinite(x).any() and np.nanmax(x) > 0:
        x = x / np.nanmax(x)
    else:
        xlabel = "Distance h (km)"

    nug, sill, a_km = _fit_spherical_quick(centers_km, gamma_vals, counts)
    x_smooth = np.linspace(x.min() if np.isfinite(x).any() else 0.0,
                           x.max() if np.isfinite(x).any() else 1.0, 200)
    hmax = np.nanmax(centers_km) if np.isfinite(centers_km).any() else 1.0
    h_smooth = x_smooth * (hmax if normalize_lag else 1.0)
    y_fit = _spherical_model(h_smooth, nug, sill, a_km)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    axN = ax.twinx()
    barw = (x[1] - x[0]) * 0.85 if x.size > 1 else 0.05
    axN.bar(x, counts, width=barw, color="#e74c3c", alpha=0.35, edgecolor="none")
    axN.set_ylabel("N", color="#e74c3c")
    axN.tick_params(axis='y', colors="#e74c3c")

    ok = np.isfinite(gamma_vals)
    ax.scatter(x[ok], gamma_vals[ok], s=18, color="#1f77b4", label="empirical γ(h)", zorder=3)
    ax.plot(x_smooth, y_fit, color="#2ecc71", lw=2.0, label="spherical fit", zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("semivariance γ(h)")
    if ylims is not None:
        ax.set_ylim(ylims[0], ylims[1])

    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left",
              bbox_to_anchor=(0.02, 0.98),
              frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.7")
    Path(outpng).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpng, dpi=140)
    plt.close(fig)


def _plot_acf_pacf_paper(ts_len, lags, acf_g, pacf_g, label_g,
                         acf_p=None, pacf_p=None, label_p=None,
                         title="", outpng=""):
    k = int(lags)
    conf = 1.96 / max(1, np.sqrt(ts_len))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True, sharey=False)
    ax = axes[0, 0]
    ax.bar(np.arange(len(acf_g)), acf_g, width=0.8)
    ax.fill_between([0, len(acf_g) - 1], conf, -conf, color="#1f77b4", alpha=0.15)
    ax.set_title(f"{label_g} ACF")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.25)
    ax = axes[0, 1]
    ax.bar(np.arange(len(pacf_g)), pacf_g, width=0.8)
    ax.fill_between([0, len(pacf_g) - 1], conf, -conf, color="#1f77b4", alpha=0.15)
    ax.set_title(f"{label_g} PACF")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.25)
    if acf_p is not None and pacf_p is not None:
        ax = axes[1, 0]
        ax.bar(np.arange(len(acf_p)), acf_p, width=0.8)
        ax.fill_between([0, len(acf_p) - 1], conf, -conf, color="#1f77b4", alpha=0.15)
        ax.set_title(f"{label_p} ACF")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.25)
        ax = axes[1, 1]
        ax.bar(np.arange(len(pacf_p)), pacf_p, width=0.8)
        ax.fill_between([0, len(pacf_p) - 1], conf, -conf, color="#1f77b4", alpha=0.15)
        ax.set_title(f"{label_p} PACF")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.25)
    for r in range(2):
        for c in range(2):
            axes[r, c].set_xlim(0, k)
            axes[r, c].set_xlabel("lag (days)")
    fig.suptitle(title, y=0.98, fontsize=13)
    Path(outpng).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpng, dpi=140)
    plt.close(fig)


def _load_area_grid_and_proc(area_dir: Path):
    _attach_project_to_syspath(area_dir.parent.parent)
    from util_tools.fixed_transfer_data_loader_old import data_processer
    from train_ddpm_with_transfer_halo_hann import get_area_data
    area_id = int(area_dir.name[-1])
    g_data, _m, _ele, (G_lats, G_lons), _days, _mall = get_area_data(area_id)
    G_lats = np.asarray(G_lats)
    G_lons = np.asarray(G_lons)
    proc = data_processer()
    return G_lats, G_lons, area_id, proc


def _load_native_sequences(args, area_dir: Path,
                           G_lats: np.ndarray, G_lons: np.ndarray,
                           area_id: int, proc):
    names = REGION_SHAPES_AFG if int(area_id) == 0 else REGION_SHAPES_GULF
    shapes = [os.path.join(args.shapes_root, n) for n in names]
    g_data, m_data, [G_lats0, G_lons0, M_lats, M_lons], _ele = proc.load_data(
        "DUEXTTAU", args.g5nr_nc, args.merra2_nc, args.elev_npy, shapes, normalize=False
    )
    g_data = np.asarray(g_data, np.float32)
    m_on_g = proc.unify_m_data(g_data[:10], m_data, G_lats, G_lons, M_lats, M_lons)
    m_on_g = np.asarray(m_on_g, np.float32)
    T = g_data.shape[0]
    dates = [str((pd.Timestamp("2005-05-15") + pd.Timedelta(days=d)).date()) for d in range(T)]
    return dates, g_data, m_on_g


def _load_id_pred_and_truth(area_dir: Path, proc, shapes_root: str):
    pred_p = area_dir / "downscaled_mean_strict.npy"
    days_p = area_dir / "pred_days_strict.npy"
    if not pred_p.exists():
        raise FileNotFoundError(f"Missing {pred_p}")
    if not days_p.exists():
        raise FileNotFoundError(f"Missing {days_p}")
    Y = np.load(pred_p).astype(np.float32)
    days = np.load(days_p)
    dates = _days_to_dates_strict(days)

    norm = np.load(area_dir / "norm_params.npz")
    gmin, gmax = float(norm["g_min"]), float(norm["g_max"])
    y_min, y_max = float(np.nanmin(Y)), float(np.nanmax(Y))
    if y_min >= -1e-6 and y_max <= 1.2:
        Y = Y * (gmax - gmin) + gmin

    g5nr_nc = "/project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc"
    merra_nc = "/project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc"
    elev = "/project/def-mere/merra2/g5nr/elevation/elevation_data.npy"

    names = REGION_SHAPES_AFG if area_dir.name.endswith("0") else REGION_SHAPES_GULF
    shapes = [os.path.join(shapes_root, n) for n in names]
    g_data, _m_data, _grids, _ele = proc.load_data("DUEXTTAU", g5nr_nc, merra_nc, elev, shapes, normalize=False)
    g_data = np.asarray(g_data, np.float32)

    T = g_data.shape[0]
    full_dates = [str((pd.Timestamp("2005-05-15") + pd.Timedelta(days=d)).date()) for d in range(T)]
    date_to_idx = {d: i for i, d in enumerate(full_dates)}
    idx = [date_to_idx[d] for d in dates if d in date_to_idx]
    G_truth = g_data[idx]
    return dates, Y, G_truth


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rundir = Path(args.rundir).resolve()
    area_dir = rundir / f"Season{args.season}" / f"Area{args.area}"
    if not area_dir.is_dir():
        raise FileNotFoundError(f"Missing {area_dir}")

    outdir = Path(args.outdir).resolve()
    _attach_project_to_syspath(rundir)
    from util_tools.fixed_transfer_data_loader_old import data_processer

    G_lats, G_lons, area_id, proc = _load_area_grid_and_proc(area_dir)

    region_mask = _region_mask(area_id, G_lats, G_lons, args.shapes_root) if args.mask else None

    def _gather_ymin_ymax(pairs_list):
        ys = []
        for _, gam, _ in pairs_list:
            if gam is None:
                continue
            g = np.asarray(gam, float)
            g = g[np.isfinite(g)]
            if g.size:
                ys.append((np.nanmin(g), np.nanmax(g)))
        if not ys:
            return None
        lo = min(a for a, _ in ys)
        hi = max(b for _, b in ys)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if hi <= lo:
            hi = lo + 1e-9
        return float(lo), float(hi)

    if args.mode == "native":
        dates_all, G, M = _load_native_sequences(args, area_dir, G_lats, G_lons, area_id, proc)
        if args.dates:
            idx = [i for i, d in enumerate(dates_all) if d in set(args.dates)]
            dates = [dates_all[i] for i in idx]
        else:
            idx = _pick_indices(len(dates_all), args.n_days, args.sample, args.seed)
            dates = [dates_all[i] for i in idx]

        all_pairs = []
        gm_pairs_per_day = []
        for i, d in zip(idx, dates):
            g = _mask_apply(G[i], region_mask)
            m = _mask_apply(M[i], region_mask)
            cg, gg, ng = empirical_variogram(g, G_lats, G_lons, region_mask,
                                             args.max_km, args.nbins, args.pairs, seed=hash(("G", d)) % 2 ** 31)
            cm, gm, nm = empirical_variogram(m, G_lats, G_lons, region_mask,
                                             args.max_km, args.nbins, args.pairs, seed=hash(("M", d)) % 2 ** 31)
            gm_pairs_per_day.append((d, (cg, gg, ng), (cm, gm, nm)))
            all_pairs += [(cg, gg, ng), (cm, gm, nm)]

        common_ylims = _gather_ymin_ymax(all_pairs) if args.sync_y == "all" else None

        for d, (cg, gg, ng), (cm, gm, nm) in gm_pairs_per_day:
            pair_ylims = _gather_ymin_ymax([(cg, gg, ng), (cm, gm, nm)]) if args.sync_y == "pair" else None
            ylims = common_ylims if args.sync_y == "all" else pair_ylims

            _plot_variogram_paper(cg, gg, ng,
                                  f"S{args.season}A{args.area}  {d}  G5NR (native)",
                                  outdir / f"semivar_native_g5nr_{d}.png",
                                  ylims=ylims)
            _plot_variogram_paper(cm, gm, nm,
                                  f"S{args.season}A{args.area}  {d}  MERRA-2→G (native)",
                                  outdir / f"semivar_native_merra_{d}.png",
                                  ylims=ylims)

        if args.semivar_all:
            cA, gA, nA = aggregate_variogram_over_days(G, G_lats, G_lons, region_mask,
                                                       args.max_km, args.nbins, args.pairs, seed=1)
            _plot_variogram_paper(cA, gA, nA,
                                  f"S{args.season}A{args.area}  ALL  G5NR (native)",
                                  outdir / "semivar_native_g5nr_ALL.png")
            cB, gB, nB = aggregate_variogram_over_days(M, G_lats, G_lons, region_mask,
                                                       args.max_km, args.nbins, args.pairs, seed=2)
            _plot_variogram_paper(cB, gB, nB,
                                  f"S{args.season}A{args.area}  ALL  MERRA-2→G (native)",
                                  outdir / "semivar_native_merra_ALL.png")

        if args.acf_pacf:
            g_ts = np.array([_nanmean2d(_mask_apply(G[i], region_mask)) for i in range(G.shape[0])], float)
            m_ts = np.array([_nanmean2d(_mask_apply(M[i], region_mask)) for i in range(M.shape[0])], float)
            T = int(G.shape[0])
            acf_g = acf_1d(g_ts, args.acf_lags)
            pacf_g = pacf_yw(g_ts, args.acf_lags)
            acf_m = acf_1d(m_ts, args.acf_lags)
            pacf_m = pacf_yw(m_ts, args.acf_lags)
            L = min(len(acf_g), len(pacf_g), len(acf_m), len(pacf_m)) - 1
            _plot_acf_pacf_paper(T, L,
                                 acf_g[:L + 1], pacf_g[:L + 1], "G5NR",
                                 acf_m[:L + 1], pacf_m[:L + 1], "MERRA-2",
                                 title=f"S{args.season}A{args.area} (native)",
                                 outpng=str(outdir / "acf_pacf_native_paper.png"))

    elif args.mode == "id":
        dates_pred, Y_pred, G_truth = _load_id_pred_and_truth(area_dir, proc, args.shapes_root)

        if args.dates:
            use = [d for d in args.dates if d in set(dates_pred)]
            idx = [dates_pred.index(d) for d in use]
            dates = use
        else:
            idx = _pick_indices(len(dates_pred), args.n_days, args.sample, args.seed)
            dates = [dates_pred[i] for i in idx]

        all_pairs = []
        yp_gp_pairs = []
        for i, d in zip(idx, dates):
            yp = _mask_apply(Y_pred[i], region_mask)
            gt = _mask_apply(G_truth[i], region_mask)
            cp, gp, np_ = empirical_variogram(yp, G_lats, G_lons, region_mask,
                                              args.max_km, args.nbins, args.pairs, seed=hash(("P", d)) % 2 ** 31)
            cg, gg, ng = empirical_variogram(gt, G_lats, G_lons, region_mask,
                                             args.max_km, args.nbins, args.pairs, seed=hash(("G", d)) % 2 ** 31)
            yp_gp_pairs.append((d, (cg, gg, ng), (cp, gp, np_)))
            all_pairs += [(cg, gg, ng), (cp, gp, np_)]

        common_ylims = _gather_ymin_ymax(all_pairs) if args.sync_y == "all" else None

        for d, (cg, gg, ng), (cp, gp, np_) in yp_gp_pairs:
            pair_ylims = _gather_ymin_ymax([(cg, gg, ng), (cp, gp, np_)]) if args.sync_y == "pair" else None
            ylims = common_ylims if args.sync_y == "all" else pair_ylims

            _plot_variogram_paper(cg, gg, ng,
                                  f"S{args.season}A{args.area}  {d}  G5NR (truth, ID dates)",
                                  outdir / f"semivar_id_g5nr_{d}.png",
                                  ylims=ylims)
            _plot_variogram_paper(cp, gp, np_,
                                  f"S{args.season}A{args.area}  {d}  Prediction (ID)",
                                  outdir / f"semivar_id_pred_{d}.png",
                                  ylims=ylims)

        if args.semivar_all:
            cG, gG, nG = aggregate_variogram_over_days(G_truth, G_lats, G_lons, region_mask,
                                                       args.max_km, args.nbins, args.pairs, seed=3)
            _plot_variogram_paper(cG, gG, nG,
                                  f"S{args.season}A{args.area}  ALL  G5NR (truth, ID dates)",
                                  outdir / "semivar_id_g5nr_ALL.png")
            cP, gP, nP = aggregate_variogram_over_days(Y_pred, G_lats, G_lons, region_mask,
                                                       args.max_km, args.nbins, args.pairs, seed=4)
            _plot_variogram_paper(cP, gP, nP,
                                  f"S{args.season}A{args.area}  ALL  Prediction (ID)",
                                  outdir / "semivar_id_pred_ALL.png")

        if args.acf_pacf:
            yp_ts = np.array([_nanmean2d(_mask_apply(Y_pred[i], region_mask)) for i in range(Y_pred.shape[0])], float)
            gt_ts = np.array([_nanmean2d(_mask_apply(G_truth[i], region_mask)) for i in range(G_truth.shape[0])], float)
            acf_p = acf_1d(yp_ts, args.acf_lags)
            pacf_p = pacf_yw(yp_ts, args.acf_lags)
            acf_g = acf_1d(gt_ts, args.acf_lags)
            pacf_g = pacf_yw(gt_ts, args.acf_lags)
            L = min(len(acf_p), len(pacf_p), len(acf_g), len(pacf_g)) - 1
            T = int(min(Y_pred.shape[0], G_truth.shape[0]))
            _plot_acf_pacf_paper(T, L,
                                 acf_g[:L + 1], pacf_g[:L + 1], "G5NR",
                                 acf_p[:L + 1], pacf_p[:L + 1], "Prediction",
                                 title=f"S{args.season}A{args.area} (ID)",
                                 outpng=str(outdir / "acf_pacf_id_paper.png"))

    else:
        pred_days = np.load(area_dir / "pred_days_strict.npy")
        id_dates = _days_to_dates_strict(pred_days)

        dates_all, G, M = _load_native_sequences(args, area_dir, G_lats, G_lons, area_id, proc)
        date_to_idx = {d: i for i, d in enumerate(dates_all)}
        idx = [date_to_idx[d] for d in id_dates if d in date_to_idx]
        dates = [dates_all[i] for i in idx]

        all_pairs = []
        gm_pairs_per_day = []
        for i, d in zip(idx, dates):
            g = _mask_apply(G[i], region_mask)
            m = _mask_apply(M[i], region_mask)
            cg, gg, ng = empirical_variogram(g, G_lats, G_lons, region_mask,
                                             args.max_km, args.nbins, args.pairs, seed=hash(("G", d)) % 2 ** 31)
            cm, gm, nm = empirical_variogram(m, G_lats, G_lons, region_mask,
                                             args.max_km, args.nbins, args.pairs, seed=hash(("M", d)) % 2 ** 31)
            gm_pairs_per_day.append((d, (cg, gg, ng), (cm, gm, nm)))
            all_pairs += [(cg, gg, ng), (cm, gm, nm)]

        common_ylims = _gather_ymin_ymax(all_pairs) if args.sync_y == "all" else None

        for d, (cg, gg, ng), (cm, gm, nm) in gm_pairs_per_day:
            pair_ylims = _gather_ymin_ymax([(cg, gg, ng), (cm, gm, nm)]) if args.sync_y == "pair" else None
            ylims = common_ylims if args.sync_y == "all" else pair_ylims

            _plot_variogram_paper(cg, gg, ng,
                                  f"",
                                  outdir / f"semivar_idnative_g5nr_{d}.png",
                                  ylims=ylims)
            _plot_variogram_paper(cm, gm, nm,
                                  f"",
                                  outdir / f"semivar_idnative_merra_{d}.png",
                                  ylims=ylims)

        if args.semivar_all and len(idx) > 0:
            cA, gA, nA = aggregate_variogram_over_days(G[idx], G_lats, G_lons, region_mask,
                                                       args.max_km, args.nbins, args.pairs, seed=11)
            _plot_variogram_paper(cA, gA, nA,
                                  f"",
                                  outdir / "semivar_idnative_g5nr_ALL.png")
            cB, gB, nB = aggregate_variogram_over_days(M[idx], G_lats, G_lons, region_mask,
                                                       args.max_km, args.nbins, args.pairs, seed=12)
            _plot_variogram_paper(cB, gB, nB,
                                  f"",
                                  outdir / "semivar_idnative_merra_ALL.png")

        if args.acf_pacf and len(idx) > 0:
            g_ts = np.array([_nanmean2d(_mask_apply(G[i], region_mask)) for i in idx], float)
            m_ts = np.array([_nanmean2d(_mask_apply(M[i], region_mask)) for i in idx], float)
            T = int(len(idx))
            acf_g = acf_1d(g_ts, args.acf_lags)
            pacf_g = pacf_yw(g_ts, args.acf_lags)
            acf_m = acf_1d(m_ts, args.acf_lags)
            pacf_m = pacf_yw(m_ts, args.acf_lags)
            L = min(len(acf_g), len(pacf_g), len(acf_m), len(pacf_m)) - 1
            _plot_acf_pacf_paper(T, L,
                                 acf_g[:L + 1], pacf_g[:L + 1], "G5NR",
                                 acf_m[:L + 1], pacf_m[:L + 1], "MERRA-2",
                                 title=f"",
                                 outpng=str(outdir / "acf_pacf_idnative_paper.png"))

    print(f"[OK] Wrote outputs → {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
