#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 5 — Temporal dependence audit (LOCKED DESIGN)

Consistent with Steps 1–4:
- Build cos(lat) area-weighted regional mean daily series first (land via NaNs)
- Do this for RAW and QDM with strict pairing
- Baseline 1985–2014, future windows: 2031–2060, 2041–2070, 2071–2100
- ERA5 provides the fixed P95 reference (regional scalar, computed from baseline)

Metrics:
A) Autocorrelation lag 1..7 of deseasonalized daily anomalies:
   anomaly(t) = T_reg(t) - climatology(doy)
   ACF(k) = corr(anom(t), anom(t-k))

B) Run-length distributions of exceedances:
   - runs of (T_reg > 40C)
   - runs of (T_reg > ERA5 P95 scalar)
   For each threshold we record:
     mean_run_length, max_run_length, runs_per_year, exceed_days_per_year

Outputs:
ANALYSIS_STEP5/step5_temporal_by_model.csv
ANALYSIS_STEP5/step5_temporal_ensemble_summary.csv
ANALYSIS_STEP5/step5_rank_and_uncertainty.csv
ANALYSIS_STEP5/step5_missing_or_errors.csv
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

OUTDIR = os.path.join(BASE, "ANALYSIS_STEP5")
os.makedirs(OUTDIR, exist_ok=True)

OUT_BYMODEL = os.path.join(OUTDIR, "step5_temporal_by_model.csv")
OUT_ENS     = os.path.join(OUTDIR, "step5_temporal_ensemble_summary.csv")
OUT_RANK    = os.path.join(OUTDIR, "step5_rank_and_uncertainty.csv")
OUT_MISS    = os.path.join(OUTDIR, "step5_missing_or_errors.csv")

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

LAGS = list(range(1, 8))
THR40 = 40.0  # C

# -------------------------
# Robust helpers (same style as Step4)
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
    tname = _find_time_var(ds)
    for v in list(ds.data_vars):
        if tname in ds[v].dims:
            return v
    raise ValueError("Could not identify temperature variable (tas/t2m).")

def to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = (da.attrs.get("units", "") or "").lower()
    if units in ["k", "kelvin"]:
        return da - 273.15
    try:
        sample = float(da.isel({da.dims[0]: 0}).mean(skipna=True).values)
        if sample > 100:
            return da - 273.15
    except Exception:
        pass
    return da

def normalize_to_standard(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
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

def year_select(ts: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    yrs = ts["time"].dt.year
    return ts.where((yrs >= y0) & (yrs <= y1), drop=True)

def area_weighted_mean_over_land(da: xr.DataArray) -> xr.DataArray:
    lat = da["latitude"]
    w = np.cos(np.deg2rad(lat)).broadcast_like(da)
    valid = da.notnull()
    num = (da * w).where(valid).sum(dim=("latitude","longitude"), skipna=True)
    den = w.where(valid).sum(dim=("latitude","longitude"), skipna=True)
    return num / den

def open_regional_series(path: str) -> xr.DataArray:
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(path, decode_times=time_coder)
    vname = find_varname(ds)
    da = normalize_to_standard(ds[vname], ds)
    da = to_celsius(da)
    return area_weighted_mean_over_land(da)

# -------------------------
# Temporal metrics
# -------------------------
def daily_doy(ts: xr.DataArray) -> xr.DataArray:
    # works for cftime and datetime
    return ts["time"].dt.dayofyear

def deseasonalize(ts: xr.DataArray) -> xr.DataArray:
    doy = daily_doy(ts)
    clim = ts.groupby(doy).mean("time", skipna=True)
    anom = ts.groupby(doy) - clim
    return anom

def acf_lags(anom: np.ndarray, lags) -> dict:
    out = {}
    x = np.asarray(anom, dtype=float)
    ok = np.isfinite(x)
    x = x[ok]
    if x.size < 20:
        for k in lags:
            out[f"acf_lag{k}"] = np.nan
        return out
    x = x - np.nanmean(x)
    for k in lags:
        if x.size <= k + 5:
            out[f"acf_lag{k}"] = np.nan
            continue
        a = x[k:]
        b = x[:-k]
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            out[f"acf_lag{k}"] = np.nan
        else:
            out[f"acf_lag{k}"] = float(np.corrcoef(a, b)[0, 1])
    return out

def run_lengths(bool_1d: np.ndarray) -> np.ndarray:
    x = np.asarray(bool_1d, dtype=bool)
    if x.size == 0:
        return np.array([], dtype=int)
    padded = np.r_[False, x, False]
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).astype(int)

def run_stats_per_year(ts: xr.DataArray, thr: float) -> dict:
    """
    Compute run-length stats for exceedances ts>thr, averaged across years.
    """
    years = np.unique(ts["time"].dt.year.values)
    per_year = []
    for yy in years:
        yts = ts.where(ts["time"].dt.year == yy, drop=True)
        if yts.sizes.get("time", 0) == 0:
            continue
        b = (yts.values > thr)
        lens = run_lengths(b)
        lens = lens[lens > 0]
        n_runs = int(lens.size)
        exceed_days = int(lens.sum()) if n_runs > 0 else 0
        mean_len = float(lens.mean()) if n_runs > 0 else 0.0
        max_len = int(lens.max()) if n_runs > 0 else 0
        per_year.append((n_runs, exceed_days, mean_len, max_len))

    if len(per_year) == 0:
        return dict(runs_per_year=np.nan, exceed_days_per_year=np.nan,
                    mean_run_length=np.nan, max_run_length=np.nan)

    arr = np.array(per_year, dtype=float)
    return dict(
        runs_per_year=float(np.nanmean(arr[:, 0])),
        exceed_days_per_year=float(np.nanmean(arr[:, 1])),
        mean_run_length=float(np.nanmean(arr[:, 2])),
        max_run_length=float(np.nanmax(arr[:, 3])),
    )

# -------------------------
# Ranking helpers
# -------------------------
def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def iqr(x):
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

# -------------------------
# File paths
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
# ERA5 P95 scalar (regional) from baseline
# -------------------------
print("Computing ERA5 regional P95 threshold (baseline 1985–2014)...")
ds_era5 = xr.open_dataset(ERA5_FILE, decode_times=True)
da_era5 = normalize_to_standard(ds_era5[ERA5_VAR], ds_era5)
da_era5 = to_celsius(da_era5)
era5_reg = area_weighted_mean_over_land(da_era5)
era5_base = year_select(era5_reg, BASELINE[0], BASELINE[1])
THR95 = float(era5_base.quantile(0.95, dim="time", skipna=True).values)
print(f"ERA5 regional P95 = {THR95:.3f} °C")

# -------------------------
# Main compute
# -------------------------
rows = []
miss = []

def compute_block(ts: xr.DataArray, y0: int, y1: int) -> dict:
    sub = year_select(ts, y0, y1)
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"No data in {y0}-{y1}")
    anom = deseasonalize(sub)
    acf = acf_lags(anom.values, LAGS)
    r40 = run_stats_per_year(sub, THR40)
    r95 = run_stats_per_year(sub, THR95)
    out = {}
    out.update(acf)
    # prefix run stats to keep columns clear
    out.update({f"run40_{k}": v for k, v in r40.items()})
    out.update({f"run95_{k}": v for k, v in r95.items()})
    return out

for model in MODELS:
    rh = path_raw_hist(model)
    qh = path_qdm_hist(model)

    if not os.path.isfile(rh):
        miss.append([model, "hist", f"missing RAW HIST: {rh}"])
        continue
    if not os.path.isfile(qh):
        miss.append([model, "hist", f"missing QDM HIST: {qh}"])
        continue

    try:
        ts_raw_hist = open_regional_series(rh)
        ts_qdm_hist = open_regional_series(qh)
    except Exception as e:
        miss.append([model, "hist", f"error opening HIST: {e}"])
        continue

    # baseline diagnostics (not deltas; audit compares structure too)
    try:
        base_raw = compute_block(ts_raw_hist, BASELINE[0], BASELINE[1])
        base_qdm = compute_block(ts_qdm_hist, BASELINE[0], BASELINE[1])
        rows.append({"model": model, "ssp": "hist", "window": "1985-2014", "stream": "RAW", **base_raw})
        rows.append({"model": model, "ssp": "hist", "window": "1985-2014", "stream": "QDM", **base_qdm})
    except Exception as e:
        miss.append([model, "hist", f"error baseline block: {e}"])

    for ssp in SSPS:
        rf = path_raw_ssp(model, ssp)
        qf = path_qdm_ssp(model, ssp)

        if not os.path.isfile(rf):
            miss.append([model, ssp, f"missing RAW {ssp}: {rf}"])
            continue
        if not os.path.isfile(qf):
            miss.append([model, ssp, f"missing QDM {ssp}: {qf}"])
            continue

        try:
            ts_raw = open_regional_series(rf)
            ts_qdm = open_regional_series(qf)
        except Exception as e:
            miss.append([model, ssp, f"error opening SSP: {e}"])
            continue

        for wlab, (y0, y1) in WINDOWS.items():
            try:
                out_raw = compute_block(ts_raw, y0, y1)
                out_qdm = compute_block(ts_qdm, y0, y1)

                rows.append({"model": model, "ssp": ssp, "window": wlab, "stream": "RAW", **out_raw})
                rows.append({"model": model, "ssp": ssp, "window": wlab, "stream": "QDM", **out_qdm})
            except Exception as e:
                miss.append([model, ssp, f"error window={wlab}: {e}"])

df = pd.DataFrame(rows)
df.to_csv(OUT_BYMODEL, index=False)

miss_df = pd.DataFrame(miss, columns=["model","ssp_or_hist","issue"])
miss_df.to_csv(OUT_MISS, index=False)

# -------------------------
# Ensemble summaries + rank diagnostics (RAW vs QDM) per variable
# -------------------------
ens_rows = []
rank_rows = []

if len(df) > 0:
    metric_cols = [c for c in df.columns if c not in ["model","ssp","window","stream"]]

    for ssp in ["hist"] + SSPS:
        for wlab in (["1985-2014"] if ssp == "hist" else list(WINDOWS.keys())):
            sub = df[(df["ssp"] == ssp) & (df["window"] == wlab)].copy()
            if len(sub) == 0:
                continue

            # paired by model
            raw = sub[sub["stream"] == "RAW"].set_index("model")
            qdm = sub[sub["stream"] == "QDM"].set_index("model")
            common = raw.index.intersection(qdm.index)
            raw = raw.loc[common]
            qdm = qdm.loc[common]
            if len(common) == 0:
                continue

            for mcol in metric_cols:
                x = raw[mcol].to_numpy(dtype=float)
                y = qdm[mcol].to_numpy(dtype=float)

                ens_rows.append({
                    "ssp": ssp,
                    "window": wlab,
                    "metric": mcol,
                    "n_models_paired": int(len(common)),
                    "raw_mean": float(np.nanmean(x)),
                    "qdm_mean": float(np.nanmean(y)),
                    "raw_iqr": iqr(x),
                    "qdm_iqr": iqr(y),
                    "mean_diff_qdm_minus_raw": float(np.nanmean(y - x)),
                    "iqr_ratio_qdm_over_raw": (iqr(y) / iqr(x)) if iqr(x) != 0 else np.nan,
                })

                # ranking diagnostics on level values
                rho = spearman(x, y)
                r_raw = pd.Series(x, index=common).rank(ascending=False, method="average")
                r_qdm = pd.Series(y, index=common).rank(ascending=False, method="average")
                abs_shift = (r_qdm - r_raw).abs()
                top3_raw = set(r_raw.nsmallest(3).index)
                top3_qdm = set(r_qdm.nsmallest(3).index)
                omega3 = len(top3_raw.intersection(top3_qdm)) / 3.0

                rank_rows.append({
                    "ssp": ssp,
                    "window": wlab,
                    "metric": mcol,
                    "spearman_rho_RAW_vs_QDM": float(rho),
                    "mean_abs_rank_shift": float(abs_shift.mean()),
                    "max_abs_rank_shift": float(abs_shift.max()),
                    "top3_overlap_Omega3": float(omega3),
                })

ens_df = pd.DataFrame(ens_rows)
ens_df.to_csv(OUT_ENS, index=False)

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(OUT_RANK, index=False)

print("=== STEP 5 COMPLETE ===")
print("By-model:", OUT_BYMODEL)
print("Ensemble:", OUT_ENS)
print("Rank/IQR:", OUT_RANK)
print("Missing:", OUT_MISS)

if not miss_df.empty:
    print("\nMissing/errors (first 20):")
    print(miss_df.head(20).to_string(index=False))