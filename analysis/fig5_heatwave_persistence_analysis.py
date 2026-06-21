#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 4 — Heatwave persistence metrics (LOCKED DESIGN)

Consistent with Steps 1–3:
- Aggregate to cos(lat) area-weighted regional mean daily series FIRST
- Land-only via NaNs (masked files)
- Then detect heatwave events on 1D daily series

Heatwaves:
- HW40: >=3 consecutive days with T_reg > 40C
- HW95: >=5 consecutive days with T_reg > ERA5 baseline P95 of regional mean (scalar)

Metrics (per period window):
- event_frequency_per_year
- mean_duration_days
- max_duration_days
- total_hw_days_per_year

Outputs:
ANALYSIS_STEP4/step4_heatwave_by_model.csv
ANALYSIS_STEP4/step4_heatwave_ensemble_summary.csv
ANALYSIS_STEP4/step4_rank_and_uncertainty.csv
ANALYSIS_STEP4/step4_missing_or_errors.csv
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

# -------------------------
# Paths / locked setup
# -------------------------
BASE = r"G:\Paper2_updated\LAND_MASKED"

RAW_HIST_DIR = os.path.join(BASE, "RAW", "HIST")
RAW_SSP_DIRS = {
    "ssp126": os.path.join(BASE, "RAW", "SSP126"),
    "ssp245": os.path.join(BASE, "RAW", "SSP245"),
    "ssp585": os.path.join(BASE, "RAW", "SSP585"),
}
QDM_HIST_DIR = os.path.join(BASE, "BC", "HIST")
QDM_SSP_DIRS = {
    "ssp126": os.path.join(BASE, "BC", "SSP126"),
    "ssp245": os.path.join(BASE, "BC", "SSP245"),
    "ssp585": os.path.join(BASE, "BC", "SSP585"),
}

ERA5_FILE = os.path.join(BASE, "ERA5", "ERA5_t2m_AOI_1985_2014_noleap_SA_LAND.nc")
ERA5_VAR = "t2m"

OUTDIR = os.path.join(BASE, "ANALYSIS_STEP4")
os.makedirs(OUTDIR, exist_ok=True)

OUT_BYMODEL = os.path.join(OUTDIR, "step4_heatwave_by_model.csv")
OUT_ENS     = os.path.join(OUTDIR, "step4_heatwave_ensemble_summary.csv")
OUT_RANK    = os.path.join(OUTDIR, "step4_rank_and_uncertainty.csv")
OUT_MISS    = os.path.join(OUTDIR, "step4_missing_or_errors.csv")

MODELS = [
    "BCC-CSM2-MR","CESM2","CMCC-ESM2","CNRM-CM6-1",
    "CNRM-ESM2-1","GFDL-ESM4","MIROC6","MRI-ESM2-0","NorESM2-MM",
]
SSPS = ["ssp126","ssp245","ssp585"]

BASELINE = (1985, 2014)
WINDOWS = {
    "2031-2060": (2031, 2060),
    "2041-2070": (2041, 2070),
    "2071-2100": (2071, 2100),
}

# Heatwave definitions (locked)
HW40_THRESH_C = 40.0
HW40_MINLEN = 3
HW95_MINLEN = 5

# -------------------------
# Helpers (from your Step1/2 philosophy)
# -------------------------
def _find_coord_var(ds: xr.Dataset, kind: str) -> str:
    assert kind in ("lat", "lon")
    target_std = "latitude" if kind == "lat" else "longitude"
    target_units = "degrees_north" if kind == "lat" else "degrees_east"

    for name, v in ds.variables.items():
        if str(v.attrs.get("standard_name", "")).lower() == target_std:
            return name
    for name, v in ds.variables.items():
        if str(v.attrs.get("units", "")).lower() == target_units:
            return name

    cand = []
    for name in ds.variables:
        low = name.lower()
        if kind == "lat" and "lat" in low:
            cand.append(name)
        if kind == "lon" and "lon" in low:
            cand.append(name)
    if cand:
        pref = ["lat","latitude"] if kind == "lat" else ["lon","longitude"]
        for p in pref:
            for c in cand:
                if c.lower() == p:
                    return c
        return cand[0]

    raise ValueError(f"Could not detect {kind} variable.")

def _find_time_var(ds: xr.Dataset) -> str:
    for t in ["time", "valid_time"]:
        if t in ds.variables or t in ds.coords:
            return t
    for name, v in ds.variables.items():
        if str(v.attrs.get("standard_name","")).lower() == "time":
            return name
    raise ValueError("Could not detect time variable.")

def find_varname(ds: xr.Dataset) -> str:
    for cand in ["tas", "t2m", "T2M", "air_temperature"]:
        if cand in ds.data_vars:
            return cand
    # fallback: first data var with time dim
    tname = _find_time_var(ds)
    for v in list(ds.data_vars):
        if tname in ds[v].dims:
            return v
    raise ValueError("Could not identify temperature variable (tas/t2m).")

def to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = (da.attrs.get("units", "") or "").lower()
    if units in ["k", "kelvin"]:
        return da - 273.15
    # heuristic if missing units
    try:
        sample = float(da.isel({da.dims[0]: 0}).mean(skipna=True).values)
        if sample > 100:
            return da - 273.15
    except Exception:
        pass
    return da

def normalize_to_standard(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    """
    Rename coords to: time, latitude, longitude
    Works for ERA5 (valid_time/latitude/longitude) and CMIP6 variants.
    """
    latn = _find_coord_var(ds, "lat")
    lonn = _find_coord_var(ds, "lon")
    timen = _find_time_var(ds)

    if latn not in da.coords:
        da = da.assign_coords({latn: ds[latn]})
    if lonn not in da.coords:
        da = da.assign_coords({lonn: ds[lonn]})
    if timen not in da.coords:
        da = da.assign_coords({timen: ds[timen]})

    rename = {}
    if timen != "time": rename[timen] = "time"
    if latn != "latitude": rename[latn] = "latitude"
    if lonn != "longitude": rename[lonn] = "longitude"
    if rename:
        da = da.rename(rename)
    return da

def year_select(da: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    yrs = da["time"].dt.year
    return da.where((yrs >= y0) & (yrs <= y1), drop=True)

def area_weighted_mean_over_land(da: xr.DataArray) -> xr.DataArray:
    """
    da: (time, latitude, longitude)
    returns: (time) regional mean
    """
    lat = da["latitude"]
    w = np.cos(np.deg2rad(lat)).broadcast_like(da)
    valid = da.notnull()
    num = (da * w).where(valid).sum(dim=("latitude","longitude"), skipna=True)
    den = w.where(valid).sum(dim=("latitude","longitude"), skipna=True)
    return num / den

def open_regional_series(path: str) -> xr.DataArray:
    """
    Open file, pick temperature var, normalize coords, convert to C,
    return regional mean daily time series (time).
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(path, decode_times=time_coder)
    vname = find_varname(ds)
    da = normalize_to_standard(ds[vname], ds)
    da = to_celsius(da)
    return area_weighted_mean_over_land(da)

def run_lengths(bool_1d: np.ndarray) -> np.ndarray:
    """
    Return lengths of consecutive True runs in a boolean 1D array.
    """
    x = np.asarray(bool_1d, dtype=bool)
    if x.size == 0:
        return np.array([], dtype=int)
    # pad with False at both ends
    padded = np.r_[False, x, False]
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).astype(int)

def heatwave_metrics_from_series(ts: xr.DataArray, threshold_c: float, min_len: int, y0: int, y1: int) -> dict:
    """
    Compute heatwave metrics on regional daily series, then average per year.
    """
    sub = year_select(ts, y0, y1)
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"No data in {y0}-{y1}")

    # group by year
    years = np.unique(sub["time"].dt.year.values)
    per_year = []

    for yy in years:
        yts = sub.where(sub["time"].dt.year == yy, drop=True)
        if yts.sizes.get("time", 0) == 0:
            continue

        hw_bool = (yts.values > threshold_c)
        lens = run_lengths(hw_bool)
        lens = lens[lens >= min_len]

        n_events = int(lens.size)
        total_days = int(lens.sum()) if n_events > 0 else 0
        mean_dur = float(lens.mean()) if n_events > 0 else 0.0
        max_dur = int(lens.max()) if n_events > 0 else 0

        per_year.append((n_events, total_days, mean_dur, max_dur))

    if len(per_year) == 0:
        return dict(event_frequency_per_year=np.nan,
                    total_hw_days_per_year=np.nan,
                    mean_duration_days=np.nan,
                    max_duration_days=np.nan)

    arr = np.array(per_year, dtype=float)
    return dict(
        event_frequency_per_year=float(np.nanmean(arr[:, 0])),
        total_hw_days_per_year=float(np.nanmean(arr[:, 1])),
        mean_duration_days=float(np.nanmean(arr[:, 2])),
        max_duration_days=float(np.nanmax(arr[:, 3])),
    )

def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def top3_overlap(vals_raw, vals_qdm):
    # higher = more hazard
    r_raw = pd.Series(vals_raw).rank(ascending=False, method="average")
    r_qdm = pd.Series(vals_qdm).rank(ascending=False, method="average")
    top3_raw = set(r_raw.nsmallest(3).index)
    top3_qdm = set(r_qdm.nsmallest(3).index)
    return len(top3_raw.intersection(top3_qdm)) / 3.0

def iqr(x):
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

# -------------------------
# File path builders (locked)
# -------------------------
def path_raw_hist(model):
    return os.path.join(RAW_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND.nc")

def path_qdm_hist(model):
    return os.path.join(QDM_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND_QDM.nc")

def path_raw_ssp(model, ssp):
    return os.path.join(RAW_SSP_DIRS[ssp], f"{model}_{ssp.upper()}_2015_2100_SA_LAND.nc")

def path_qdm_ssp(model, ssp):
    return os.path.join(QDM_SSP_DIRS[ssp], f"{model}_{ssp.upper()}_2015_2100_SA_LAND_QDM.nc")

# -------------------------
# Compute ERA5 P95 of REGIONAL series (baseline)
# -------------------------
print("Computing ERA5 regional P95 threshold (baseline 1985–2014)...")
ds_era5 = xr.open_dataset(ERA5_FILE, decode_times=True)
da_era5 = normalize_to_standard(ds_era5[ERA5_VAR], ds_era5)
da_era5 = to_celsius(da_era5)
era5_reg = area_weighted_mean_over_land(da_era5)
era5_base = year_select(era5_reg, BASELINE[0], BASELINE[1])

HW95_THRESH_C = float(era5_base.quantile(0.95, dim="time", skipna=True).values)
print(f"ERA5 HW95 threshold (regional P95): {HW95_THRESH_C:.3f} °C")

# -------------------------
# Main compute
# -------------------------
rows = []
miss = []

# heatwave definitions list
HW_DEFS = [
    ("HW40", HW40_THRESH_C, HW40_MINLEN),
    ("HW95", HW95_THRESH_C, HW95_MINLEN),
]

for model in MODELS:
    p_raw_hist = path_raw_hist(model)
    p_qdm_hist = path_qdm_hist(model)

    if not os.path.isfile(p_raw_hist):
        miss.append([model, "hist", f"missing RAW HIST: {p_raw_hist}"])
        continue
    if not os.path.isfile(p_qdm_hist):
        miss.append([model, "hist", f"missing QDM HIST: {p_qdm_hist}"])
        continue

    # Open baseline regional series once per stream
    try:
        ts_raw_hist = open_regional_series(p_raw_hist)
        ts_qdm_hist = open_regional_series(p_qdm_hist)
    except Exception as e:
        miss.append([model, "hist", f"error opening HIST series: {e}"])
        continue

    # Baseline metrics (1985–2014) per HW def
    base_raw = {}
    base_qdm = {}
    for hw_name, thr_c, minlen in HW_DEFS:
        try:
            base_raw[hw_name] = heatwave_metrics_from_series(ts_raw_hist, thr_c, minlen, BASELINE[0], BASELINE[1])
        except Exception as e:
            miss.append([model, "hist", f"RAW baseline {hw_name} error: {e}"])
            base_raw[hw_name] = None
        try:
            base_qdm[hw_name] = heatwave_metrics_from_series(ts_qdm_hist, thr_c, minlen, BASELINE[0], BASELINE[1])
        except Exception as e:
            miss.append([model, "hist", f"QDM baseline {hw_name} error: {e}"])
            base_qdm[hw_name] = None

    for ssp in SSPS:
        p_raw = path_raw_ssp(model, ssp)
        p_qdm = path_qdm_ssp(model, ssp)

        if not os.path.isfile(p_raw):
            miss.append([model, ssp, f"missing RAW {ssp}: {p_raw}"])
            continue
        if not os.path.isfile(p_qdm):
            miss.append([model, ssp, f"missing QDM {ssp}: {p_qdm}"])
            continue

        # paired rule: open both
        try:
            ts_raw = open_regional_series(p_raw)
            ts_qdm = open_regional_series(p_qdm)
        except Exception as e:
            miss.append([model, ssp, f"error opening SSP series: {e}"])
            continue

        for wlab, (y0, y1) in WINDOWS.items():
            for hw_name, thr_c, minlen in HW_DEFS:
                try:
                    fut_raw = heatwave_metrics_from_series(ts_raw, thr_c, minlen, y0, y1)
                    fut_qdm = heatwave_metrics_from_series(ts_qdm, thr_c, minlen, y0, y1)

                    braw = base_raw[hw_name]
                    bqdm = base_qdm[hw_name]
                    if braw is None or bqdm is None:
                        miss.append([model, ssp, f"baseline missing for {hw_name}, skipping deltas"])
                        continue

                    for metric_key in ["event_frequency_per_year","total_hw_days_per_year","mean_duration_days","max_duration_days"]:
                        rows.append({
                            "model": model,
                            "ssp": ssp,
                            "window": wlab,
                            "heatwave": hw_name,
                            "metric": metric_key,

                            "baseline_RAW": braw[metric_key],
                            "future_RAW": fut_raw[metric_key],
                            "delta_RAW": fut_raw[metric_key] - braw[metric_key],

                            "baseline_QDM": bqdm[metric_key],
                            "future_QDM": fut_qdm[metric_key],
                            "delta_QDM": fut_qdm[metric_key] - bqdm[metric_key],

                            "delta_diff_QDM_minus_RAW": (fut_qdm[metric_key] - bqdm[metric_key]) - (fut_raw[metric_key] - braw[metric_key]),
                        })
                except Exception as e:
                    miss.append([model, ssp, f"error {wlab} {hw_name}: {e}"])

df = pd.DataFrame(rows)
df.to_csv(OUT_BYMODEL, index=False)

miss_df = pd.DataFrame(miss, columns=["model","ssp_or_hist","issue"])
miss_df.to_csv(OUT_MISS, index=False)

# -------------------------
# Ensemble summary + ranking (like Step2/3)
# -------------------------
ens_rows = []
rank_rows = []

if len(df) > 0:
    for (heatwave, metric), dsub0 in df.groupby(["heatwave","metric"]):
        for ssp in SSPS:
            for wlab in WINDOWS.keys():
                sub = dsub0[(dsub0["ssp"] == ssp) & (dsub0["window"] == wlab)].copy()
                if len(sub) == 0:
                    continue

                raw = sub["delta_RAW"].to_numpy(dtype=float)
                qdm = sub["delta_QDM"].to_numpy(dtype=float)

                ens_rows.append({
                    "heatwave": heatwave,
                    "metric": metric,
                    "ssp": ssp,
                    "window": wlab,
                    "n_models": int(sub["model"].nunique()),
                    "mean_delta_RAW": float(np.nanmean(raw)),
                    "mean_delta_QDM": float(np.nanmean(qdm)),
                    "iqr_delta_RAW": iqr(raw),
                    "iqr_delta_QDM": iqr(qdm),
                    "mean_delta_diff_QDM_minus_RAW": float(np.nanmean(qdm - raw)),
                    "iqr_ratio_QDM_over_RAW": (iqr(qdm) / iqr(raw)) if iqr(raw) != 0 else np.nan,
                })

                rho = spearman(raw, qdm)

                r_raw = pd.Series(raw, index=sub["model"]).rank(ascending=False, method="average")
                r_qdm = pd.Series(qdm, index=sub["model"]).rank(ascending=False, method="average")
                abs_shift = (r_qdm - r_raw).abs()

                omega3 = len(set(r_raw.nsmallest(3).index).intersection(set(r_qdm.nsmallest(3).index))) / 3.0

                rank_rows.append({
                    "heatwave": heatwave,
                    "metric": metric,
                    "ssp": ssp,
                    "window": wlab,
                    "spearman_rho_RAW_vs_QDM": float(rho),
                    "mean_abs_rank_shift": float(abs_shift.mean()),
                    "max_abs_rank_shift": float(abs_shift.max()),
                    "top3_overlap_Omega3": float(omega3),
                })

ens_df = pd.DataFrame(ens_rows)
ens_df.to_csv(OUT_ENS, index=False)

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(OUT_RANK, index=False)

print("=== STEP 4 COMPLETE ===")
print("By-model:", OUT_BYMODEL)
print("Ensemble:", OUT_ENS)
print("Rank/IQR:", OUT_RANK)
print("Missing:", OUT_MISS)

if not miss_df.empty:
    print("\nMissing/errors (first 20):")
    print(miss_df.head(20).to_string(index=False))