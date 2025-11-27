#  Natural‑season train-test split script for training models
#  Author:  Yang Xiang, Date: 2025‑07‑28
# -------------------------------------------------------------------------
# How to use the script?
# Example: 1: When training the large model (we need two year data)
# python get_large_data_ready.py  /project/def-mere/merra2/Downscale/Result /project/def-mere/merra2/g5nr/G5NR_merged_daily_noclip_2005-2007.nc
# 2: When training the small transfer model, call the path of merra2 data with more than 2 years.

# The script will produce two npy files recording train and test indices for each season across all year ranges 
# in the data respectively, as well as a safe csv file recorded to check if the generated test train days are good.


import argparse, os, random, sys
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc
import pandas  as pd 

SEASONS = {
    "DJF": (12, 1, 2),   # winter
    "MAM": (3, 4, 5),    # spring
    "JJA": (6, 7, 8),    # summer
    "SON": (9,10,11),    # autumn
}
ORDER   = ["DJF", "MAM", "JJA", "SON"]    


def decode_netcdf_time(path:str, var:str="time"):
    """
    Return a list[datetime] for the given NetCDF 'time' variable.

    """

    with nc.Dataset(path) as ds:
        time_var = ds.variables[var]
        return nc.num2date(time_var[:], units=time_var.units)  

def season_of(date:datetime) -> str:
    """
    Determine which natural season the given date is from. Return DJF / MAM / JJA / SON for a given date.

    """

    m = date.month
    for name, months in SEASONS.items(): 
        if m in months: return name    
    raise ValueError(f"Bad month: {m}")


# Guess the student ID that the seed is set to be belongs to whom? 
def build_split(dates, *, n_lag: int, test_ratio=0.10, buffer_days=45, seed=1007715536):
    """
    Parameters
    ----------
    dates :  The list of datetime objects from decode_netcdf_time.
    test_ratio : float, the fraction of data within a season to be used for testing (0.1).
    buffer_days : int, days we prepend to the season and *allow only for training* (as in Dr. Wang's 2025 paper)
    Returns
    -------
    dict[season_name] = (train_idx:list[int], test_idx:list[int])
    IMPORTANT! Indices are *1‑based* – the rest of the pipeline expects that.

    """

    n = len(dates)  
    rng  = random.Random(seed)
    idx1 = np.arange(1, n+1)    

    core = {s: [] for s in SEASONS}  
    for i, d in zip(idx1, dates):  
        core[season_of(d)].append(i) 

    train_pool = {}
    for s in SEASONS: 
        pool = set(core[s])
        for i in core[s]:
            for j in range(1, buffer_days+1):  
                if i-j >= 1: pool.add(i-j)  
        train_pool[s] = sorted(pool)  

    split = {}
    for s in SEASONS:

        candidates = [i for i in core[s] if i > n_lag]
        test_sz    = int(len(candidates) * test_ratio)
        test_set   = set(rng.sample(candidates, test_sz))
        train_set = set(train_pool[s]) - test_set
        split[s] = (sorted(train_set), sorted(test_set))

    return split


def main():
    p = argparse.ArgumentParser(description = "Generate natural‑season train / test indices with lag safety.")
    p.add_argument("out_dir", help = "where Season*/train_days.npy will go")  

    p.add_argument("nc_path", help = "any NetCDF file with a time axis")       
    p.add_argument("--n_lag", type=int, default=40, help="minimum history length a test day must have")
    p.add_argument("--test_ratio", type = float, default = 0.10)
    p.add_argument("--buffer", type = int, default = 45, help = "days before the season core that enter the *train* pool")
    args = p.parse_args()   

    dates = decode_netcdf_time(args.nc_path)
    split = build_split(dates, n_lag = args.n_lag, test_ratio = args.test_ratio, buffer_days = args.buffer)

    os.makedirs(args.out_dir, exist_ok=True)

    for k, season in enumerate(ORDER, start=1):   
        train, test = split[season]
        season_dir  = os.path.join(args.out_dir, f"Season{k}")
        os.makedirs(season_dir, exist_ok=True)

        np.save(os.path.join(season_dir, "train_days.npy"), np.array(train))
        np.save(os.path.join(season_dir, "test_days.npy"),  np.array(test))

        df = (pd.DataFrame({"idx"  : train + test, "date" : [dates[i-1].strftime("%Y-%m-%d") for i in train+test],
        "split": ["train"]*len(train) + ["test"]*len(test)}).sort_values("idx"))
        df.to_csv(os.path.join(season_dir, "season_split.csv"), index=False)

        print(f"{season_dir}:  {len(train):5d} train | {len(test):5d} test | CSV are all saved!")
        for area in (0, 1):
            os.makedirs(os.path.join(season_dir, f"Area{area}"), exist_ok=True)

if __name__ == "__main__":
    main()
