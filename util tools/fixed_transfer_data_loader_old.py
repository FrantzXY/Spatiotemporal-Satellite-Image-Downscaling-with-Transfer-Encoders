###_______ Author: Yige Yan, Date: 2025-06-05 _______###

LAT_MIN = 10.875
LAT_MAX = 43.25
LON_MIN = 34.4375
LON_MAX = 80.25

import os
import sys
import math
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from math import exp, sqrt, log
import time
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
from typing import Tuple


class data_processer():
    def __init__(self):
        pass

    def country_filter(self,
                       sample_image,
                       lats,
                       lons,
                       country_shape,
                       return_obj=False):
        data = xr.DataArray(sample_image, dims=('y', 'x'), coords={'y': lats, 'x': lons})
        lidar_clipped = data.rio.set_crs(country_shape.crs).rio.clip(country_shape.geometry)

        if return_obj:
            return lidar_clipped, lidar_clipped.values, lidar_clipped['y'].values, lidar_clipped['x'].values
        else:
            return lidar_clipped.values, lidar_clipped['y'].values, lidar_clipped['x'].values

    def norm_per_array(self, arr: np.ndarray) -> np.ndarray:
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    def load_data(self, target_variable, file_path_g,
                  file_path_m, file_path_ele, file_path_country, normalize=True):
        country_shape = gpd.read_file(file_path_country[0])
        if len(file_path_country) > 1:
            for country_path in file_path_country[1:]:
                country_shape = pd.concat([country_shape, gpd.read_file(country_path)])
        self.countryshape = country_shape

        lonmin, latmin, lonmax, latmax = country_shape.total_bounds

        g_nc_data = nc.Dataset(file_path_g)
        g_data = g_nc_data.variables[target_variable][:]

        eps = 1e-6
        g_data = np.log10(np.maximum(g_data, eps))

        G_lats = g_nc_data.variables['lat'][:]
        G_lons = g_nc_data.variables['lon'][:]
        latmin_ind = np.argmin(np.abs(G_lats - latmin))
        latmax_ind = np.argmin(np.abs(G_lats - latmax))
        lonmin_ind = np.argmin(np.abs(G_lons - lonmin))
        lonmax_ind = np.argmin(np.abs(G_lons - lonmax))
        latmin_ind = max(0, latmin_ind - 1)
        lonmin_ind = max(0, lonmin_ind - 1)

        g_data = g_data[:, latmin_ind:latmax_ind + 1, lonmin_ind:lonmax_ind + 1]
        G_lats = g_nc_data.variables['lat'][latmin_ind:latmax_ind + 1]
        G_lons = g_nc_data.variables['lon'][lonmin_ind:lonmax_ind + 1]

        if file_path_ele is not None:
            ele_data = np.load(file_path_ele)
            ele_data = ele_data[latmin_ind:latmax_ind + 1,
                                lonmin_ind:lonmax_ind + 1]
            if normalize:
                ele_data = self.normalize(ele_data)
        else:
            ele_data = np.zeros((latmax_ind - latmin_ind + 1,
                                 lonmax_ind - lonmin_ind + 1),
                                dtype=np.float32)

        m_ncdata = nc.Dataset(file_path_m)
        m_data = m_ncdata.variables[target_variable][:, :, :]
        m_data = np.log10(np.maximum(m_data, eps))

        M_lats_full = m_ncdata.variables['lat'][:]
        M_lons_full = m_ncdata.variables['lon'][:]
        latmin_ind_m = np.argmin(np.abs(M_lats_full - latmin))
        latmax_ind_m = np.argmin(np.abs(M_lats_full - latmax))
        lonmin_ind_m = np.argmin(np.abs(M_lons_full - lonmin))
        lonmax_ind_m = np.argmin(np.abs(M_lons_full - lonmax))
        latmin_ind_m = max(0, latmin_ind_m - 1)
        lonmin_ind_m = max(0, lonmin_ind_m - 1)

        m_data = m_data[:, latmin_ind_m:latmax_ind_m + 1, lonmin_ind_m:lonmax_ind_m + 1]
        M_lats = M_lats_full[latmin_ind_m:latmax_ind_m + 1]
        M_lons = M_lons_full[lonmin_ind_m:lonmax_ind_m + 1]

        return g_data, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data

    def normalize(self, data):
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val == 0:
            print("[WARN] Zero range in normalize(), returning zeros")
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def unify_m_data(self, g_data, m_data, G_lats, G_lons, M_lats, M_lons):
        lat_dim, lon_dim = g_data.shape[1:]
        unif_m_data = np.zeros((m_data.shape[0], lat_dim, lon_dim))
        for i in range(lat_dim):
            for j in range(lon_dim):
                lat = G_lats[i]
                lon = G_lons[j]
                m_lat_idx = np.argmin(np.abs(M_lats - lat))
                m_lon_idx = np.argmin(np.abs(M_lons - lon))
                unif_m_data[:, i, j] = m_data[:, m_lat_idx, m_lon_idx]
        return unif_m_data

    def cut_country(self, g_data, m_data, lat_lon, ele_data):
        G_lats, G_lons, M_lats, M_lons = lat_lon

        if g_data.shape != m_data.shape:
            raise ValueError('G data and M data are not in a consistent shape!')
        croped_g = []
        croped_m = []
        for i in range(g_data.shape[0]):
            c_g, _, _ = self.country_filter(g_data[i], G_lats, G_lons, self.countryshape)
            c_m, _, _ = self.country_filter(m_data[i], G_lats, G_lons, self.countryshape)
            croped_g.append(c_g)
            croped_m.append(c_m)
        croped_ele, lats, lons = self.country_filter(ele_data, G_lats, G_lons, self.countryshape)
        return np.array(croped_g), np.array(croped_m), croped_ele, lats, lons

    def flatten(self,
                h_data,
                l_data,
                ele_data,
                lat_lon,
                days,
                n_lag,
                n_pred,
                task_dim,
                is_perm=True,
                return_Y=True,
                stride=1,
                return_nan=False):

        g5nr_start_day_offset = 1961
        absolute_day_indices = np.array(days) - 1 + g5nr_start_day_offset

        timestamps = pd.to_datetime(absolute_day_indices, unit='D', origin='2000-01-01')
        years = timestamps.year
        min_year, max_year = 2000, 2024
        year_norm_lookup = ((years - min_year) / (max_year - min_year)).astype(np.float32)

        task_lat_dim, task_lon_dim = task_dim
        G_lats, G_lons = lat_lon
        X_high = []
        X_low = []
        X_ele = []
        X_other = []
        Y = []

        end_point = h_data.shape[0] - n_pred if return_Y else h_data.shape[0]
        end_point = min(end_point, l_data.shape[0] - n_pred)

        for t in range(n_lag - 1, end_point):
            lat_pos = list(range(task_lat_dim, h_data.shape[1] + 1, stride))
            if lat_pos[-1] != h_data.shape[1]:
                lat_pos.append(h_data.shape[1])

            lon_pos = list(range(task_lon_dim, h_data.shape[2] + 1, stride))
            if lon_pos[-1] != h_data.shape[2]:
                lon_pos.append(h_data.shape[2])
            for lat in lat_pos:
                for lon in lon_pos:
                    lat_raw = G_lats[lat - task_lat_dim]
                    lon_raw = G_lons[lon - task_lon_dim]
                    lat_scaled = (lat_raw - LAT_MIN) / (LAT_MAX - LAT_MIN)
                    lon_scaled = (lon_raw - LON_MIN) / (LON_MAX - LON_MIN)

                    if h_data[(t - n_lag + 1):t + 1, (lat - task_lat_dim):lat,
                              (lon - task_lon_dim):lon].shape != (n_lag, task_lat_dim, task_lon_dim):
                        print('t:', (t - n_lag + 1), t + 1)
                        print('lat: ', (lat - task_lat_dim), lat)
                        print('lon: ', (lon - task_lon_dim), lon)
                    else:
                        if return_nan:
                            X_high.append(
                                h_data[(t - n_lag + 1):t + 1, (lat - task_lat_dim):lat, (lon - task_lon_dim):lon]
                            )
                            if return_Y:
                                Y.append(
                                    h_data[t + 1:(t + n_pred + 1), (lat - task_lat_dim):lat,
                                           (lon - task_lon_dim):lon]
                                )
                            low_slice = l_data[t + 1,
                                               lat - task_lat_dim:lat,
                                               lon - task_lon_dim:lon][np.newaxis, ...]
                            X_low.append(low_slice)
                            X_ele.append(ele_data[(lat - task_lat_dim):lat, (lon - task_lon_dim):lon])
                            year_norm = year_norm_lookup[t + 1]
                            X_other.append(
                                [lat_scaled, lon_scaled, (days[t + 1] % 365) / 365, year_norm]
                            )
                        else:
                            window_ok = not np.isnan(h_data[t, lat - 1, lon - 1]) \
                                and not np.isnan(
                                    h_data[(t - n_lag + 1):t + 1,
                                           (lat - task_lat_dim):lat,
                                           (lon - task_lon_dim):lon]
                                ).any()
                            if window_ok:
                                X_high.append(
                                    h_data[(t - n_lag + 1):t + 1,
                                           (lat - task_lat_dim):lat,
                                           (lon - task_lon_dim):lon]
                                )
                                if return_Y:
                                    Y.append(
                                        h_data[t + 1:(t + n_pred + 1),
                                               (lat - task_lat_dim):lat,
                                               (lon - task_lon_dim):lon]
                                    )
                                low_slice = l_data[t + 1,
                                                   lat - task_lat_dim:lat,
                                                   lon - task_lon_dim:lon][np.newaxis, ...]
                                if not np.isnan(low_slice).any():
                                    X_low.append(low_slice)
                                    X_ele.append(ele_data[(lat - task_lat_dim):lat, (lon - task_lon_dim):lon])
                                    year_norm = year_norm_lookup[t + 1]
                                    X_other.append(
                                        [lat_scaled, lon_scaled, (days[t + 1] % 365) / 365, year_norm]
                                    )
        if is_perm:
            perm = np.random.permutation(len(X_high))
            if return_Y:
                return np.expand_dims(np.array(X_high, dtype=np.float32), -1)[perm], \
                    np.expand_dims(np.array(X_low, dtype=np.float32), -1)[perm], \
                    np.expand_dims(np.array(X_ele, dtype=np.float32), -1)[perm], \
                    np.array(X_other, dtype=np.float32)[perm], \
                    np.array(Y, dtype=np.float32)[perm]
            else:
                return np.expand_dims(np.array(X_high, dtype=np.float32), -1)[perm], \
                    np.expand_dims(np.array(X_low, dtype=np.float32), -1)[perm], \
                    np.expand_dims(np.array(X_ele, dtype=np.float32), -1)[perm], \
                    np.array(X_other, dtype=np.float32)[perm]
        if return_Y:
            return np.expand_dims(np.array(X_high, dtype=np.float32), -1), \
                np.expand_dims(np.array(X_low, dtype=np.float32), -1), \
                np.expand_dims(np.array(X_ele, dtype=np.float32), -1), \
                np.array(X_other, dtype=np.float32), np.array(Y, dtype=np.float32)
        else:
            return np.expand_dims(np.array(X_high, dtype=np.float32), -1), \
                np.expand_dims(np.array(X_low, dtype=np.float32), -1), \
                np.expand_dims(np.array(X_ele, dtype=np.float32), -1), \
                np.array(X_other, dtype=np.float32)

    def flatten_transfer(
        self,
        l_data: np.ndarray,
        *,
        n_lag: int,
        n_pred: int,
        task_dim: Tuple[int, int],
        is_perm: bool = True,
        return_Y: bool = True,
        stride: int = 1,
        return_nan: bool = False,
    ):
        t_lat, t_lon = task_dim
        X_low, Y = [], []

        end_pt = l_data.shape[0] - (n_pred if return_Y else 0)

        for t in range(n_lag - 1, end_pt):
            lat_pos = list(range(t_lat, l_data.shape[1] + 1, stride))
            if lat_pos[-1] != l_data.shape[1]:
                lat_pos.append(l_data.shape[1])

            lon_pos = list(range(t_lon, l_data.shape[2] + 1, stride))
            if lon_pos[-1] != l_data.shape[2]:
                lon_pos.append(l_data.shape[2])
            for lat in lat_pos:
                for lon in lon_pos:
                    window = l_data[
                        t - n_lag + 1:t + 1, lat - t_lat:lat, lon - t_lon:lon
                    ]
                    if window.shape != (n_lag, t_lat, t_lon):
                        continue
                    if not return_nan and np.isnan(window[-1, -1, -1]):
                        continue
                    X_low.append(window)
                    if return_Y:
                        Y.append(
                            l_data[
                                t + 1:t + 1 + n_pred,
                                lat - t_lat:lat,
                                lon - t_lon:lon,
                            ]
                        )

        X_low = np.expand_dims(np.asarray(X_low, dtype=np.float32), -1)

        if is_perm:
            idx = np.random.permutation(len(X_low))
            X_low = X_low[idx]
            if return_Y:
                Y = np.asarray(Y, dtype=np.float32)[idx]

        return (X_low, np.asarray(Y, dtype=np.float32)) if return_Y else X_low


if __name__ == "__main__":
    start = time.time()