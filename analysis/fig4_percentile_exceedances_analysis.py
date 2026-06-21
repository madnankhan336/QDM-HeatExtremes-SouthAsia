#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 3 — Percentile extremes using fixed ERA5 reference (LOCKED DESIGN)

Fixes vs your current Step3:
- ERA5 variable = t2m; CMIP6 variable = tas (auto-detected like Step1/2, but prefer tas)
- ERA5 coords valid_time/latitude/longitude normalized to time/latitude/longitude
- CMIP6 coords normalized to time/latitude/longitude (robust)
- Kelvin->Celsius conversion applied consistently (ERA5 + CMIP6)
- Uses SAME aggregation logic style as Step1/2:
    spatial mean of exceedance indicator (0/1) with cos(lat) weighting over valid land cells,
    then annual sum of days, then mean across years in each window.

Outputs (matches your Step2 style):
ANALYSIS_STEP3/step3_percentile_exceedance_by_model.csv
ANALYSIS_STEP3/step3_percentile_exceedance_ensemble_summary.csv
ANALYSIS_STEP3/step3_rank_and_uncertainty.csv
ANALYSIS_STEP3/step3_missing_or_errors.csv
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

OUTDIR = os.path.join(BASE, "ANALYSIS_STEP3")
os.makedirs(OUTDIR, exist_ok=True)

OUT_BYMODEL = os.path.join(OUTDIR, "step3_percentile_exceedance_by_model.csv")
OUT_ENS     = os.path.join(OUTDIR, "step3_percentile_exceedance_ensemble_summary.csv")
OUT_RANK    = os.path.join(OUTDIR, "step3_rank_and_uncertainty.csv")
OUT_MISS    = os.path.join(OUTDIR, "step3_missing_or_errors.csv")

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

# Locked percentiles from ERA5 baseline only
PCTS = [95, 99]

# ERA5 var is t2m; CMIP6 typically tas (but we auto-detect robustly)
ERA5_VAR = "t2m"

# -------------------------
# Robust coord + var handling (aligned with your Step1/2 philosophy)
# -------------------------
def _find_time_var(ds: xr.Dataset) -> str:
    for t in ["time", "valid_time"]:
        if t in ds.variables or t in ds.coords:
            return t
    for name, v in ds.variables.items():
        if str(v.attrs.get("standard_name","")).lower() == "time":
            return name
    raise ValueError("Could not detect time variable.")

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

    # fallback by name pattern
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

def find_varname(ds: xr.Dataset) -> str:
    # Prefer CMIP6 tas; ERA5 uses t2m; otherwise any var with time dim
    for cand in ["tas", "t2m", "T2M", "air_temperature"]:
        if cand in ds.data_vars:
            return cand
    for v in list(ds.data_vars):
        if _find_time_var(ds) in ds[v].dims:
            return v
    raise ValueError("Could not identify temperature variable (tas/t2m).")

def normalize_to_standard(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    """
    Ensure da has coords named exactly: time, latitude, longitude.
    Works for ERA5 (valid_time/latitude/longitude) and CMIP6 (time/lat/lon variants).
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

def to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = (da.attrs.get("units", "") or "").lower()
    if units in ["k", "kelvin"]:
        return da - 273.15
    # heuristic if units missing
    try:
        sample = float(da.isel(time=0).mean(skipna=True).values)
        if sample > 100:
            return da - 273.15
    except Exception:
        pass
    return da

def year_select(da: xr.DataArray, y0: int, y1: int) -> xr.DataArray:
    yrs = da["time"].dt.year
    return da.where((yrs >= y0) & (yrs <= y1), drop=True)

def area_weighted_mean_over_land_indicator(exceed01: xr.DataArray) -> xr.DataArray:
    """
    exceed01: (time, latitude, longitude) float 0/1 (NaNs allowed)
    Returns: (time) area-weighted mean over VALID cells (land mask).
    """
    lat = exceed01["latitude"]
    w = np.cos(np.deg2rad(lat))

    # valid cells determined from first timestep (land mask stable)
    valid = xr.where(np.isfinite(exceed01.isel(time=0)), 1.0, np.nan)

    ww = w.broadcast_like(exceed01.isel(time=0)) * valid  # (lat,lon)
    num = (exceed01 * ww).sum(dim=("latitude", "longitude"), skipna=True)
    den = ww.sum(dim=("latitude", "longitude"), skipna=True)
    return num / den

def mean_annual_exceedance_days_from_threshold(da_c: xr.DataArray, thr2d_c: xr.DataArray, y0: int, y1: int) -> float:
    """
    da_c: temperature in degC (time, latitude, longitude)
    thr2d_c: threshold in degC (latitude, longitude) on same grid
    Returns mean annual exceedance days over years y0..y1:
      daily exceed01 -> spatial weighted mean -> annual sum -> mean over years
    """
    da_c = year_select(da_c, y0, y1)

    # broadcast threshold to time
    exceed = (da_c > thr2d_c).astype("float32")  # (time, lat, lon), NaNs propagate if da has NaNs
    reg = area_weighted_mean_over_land_indicator(exceed)  # (time)

    annual = reg.groupby(reg["time"].dt.year).sum("time")  # days/year (area-weighted mean)
    return float(annual.mean(skipna=True).values)

# Spearman without scipy (rank then Pearson)
def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

# -------------------------
# File path builders (locked)
# -------------------------
def raw_hist_file(model: str) -> str:
    return os.path.join(RAW_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND.nc")

def qdm_hist_file(model: str) -> str:
    return os.path.join(QDM_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND_QDM.nc")

def raw_ssp_file(model: str, ssp: str) -> str:
    return os.path.join(RAW_SSP_DIRS[ssp], f"{model}_{ssp.upper()}_2015_2100_SA_LAND.nc")

def qdm_ssp_file(model: str, ssp: str) -> str:
    return os.path.join(QDM_SSP_DIRS[ssp], f"{model}_{ssp.upper()}_2015_2100_SA_LAND_QDM.nc")

# -------------------------
# ERA5 thresholds (fixed baseline reference)
# -------------------------
def compute_era5_thresholds() -> dict:
    """
    Returns thresholds dict:
      {95: thr_p95(lat,lon), 99: thr_p99(lat,lon)} in degC
    """
    ds = xr.open_dataset(ERA5_FILE, decode_times=True)

    if ERA5_VAR not in ds.data_vars:
        raise ValueError(f"ERA5 missing '{ERA5_VAR}'. Found: {list(ds.data_vars)}")

    da = normalize_to_standard(ds[ERA5_VAR], ds)
    da = to_celsius(da)

    da_b = year_select(da, BASELINE[0], BASELINE[1])

    thr = {}
    for p in PCTS:
        q = da_b.quantile(p/100.0, dim="time", skipna=True)
        if "quantile" in q.dims:
            q = q.squeeze("quantile", drop=True)
        thr[p] = q
    return thr

# -------------------------
# Read CMIP6 temp (tas/t2m) normalized + degC
# -------------------------
def open_cmip6_temp(path: str) -> xr.DataArray:
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(path, decode_times=time_coder)

    vname = find_varname(ds)
    da = normalize_to_standard(ds[vname], ds)
    da = to_celsius(da)
    return da

# -------------------------
# Main computation
# -------------------------
rows = []
miss = []

print("Computing ERA5 fixed thresholds (baseline 1985–2014)...")
thr = compute_era5_thresholds()
print("Computed ERA5 fixed thresholds:", ", ".join([f"p{p}" for p in PCTS]))

for model in MODELS:
    # Baseline files must exist for both RAW and QDM (paired rule)
    p_raw_hist = raw_hist_file(model)
    p_qdm_hist = qdm_hist_file(model)

    if not os.path.isfile(p_raw_hist):
        miss.append([model, "hist", f"missing RAW HIST: {p_raw_hist}"])
        continue
    if not os.path.isfile(p_qdm_hist):
        miss.append([model, "hist", f"missing QDM HIST: {p_qdm_hist}"])
        continue

    # baseline exceedance days for P95/P99 for both streams
    base_raw = {}
    base_qdm = {}
    try:
        da_raw_hist = open_cmip6_temp(p_raw_hist)
        da_qdm_hist = open_cmip6_temp(p_qdm_hist)
    except Exception as e:
        miss.append([model, "hist", f"error opening HIST files: {e}"])
        continue

    for p in PCTS:
        try:
            base_raw[p] = mean_annual_exceedance_days_from_threshold(
                da_raw_hist, thr[p], BASELINE[0], BASELINE[1]
            )
        except Exception as e:
            miss.append([model, "hist", f"RAW HIST error p{p}: {e}"])
            base_raw[p] = np.nan

        try:
            base_qdm[p] = mean_annual_exceedance_days_from_threshold(
                da_qdm_hist, thr[p], BASELINE[0], BASELINE[1]
            )
        except Exception as e:
            miss.append([model, "hist", f"QDM HIST error p{p}: {e}"])
            base_qdm[p] = np.nan

    # futures
    for ssp in SSPS:
        p_raw = raw_ssp_file(model, ssp)
        p_qdm = qdm_ssp_file(model, ssp)

        if not os.path.isfile(p_raw):
            miss.append([model, ssp, f"missing RAW {ssp}: {p_raw}"])
            continue
        if not os.path.isfile(p_qdm):
            miss.append([model, ssp, f"missing QDM {ssp}: {p_qdm}"])
            continue

        try:
            da_raw = open_cmip6_temp(p_raw)
            da_qdm = open_cmip6_temp(p_qdm)
        except Exception as e:
            miss.append([model, ssp, f"error opening SSP files: {e}"])
            continue

        for wlab, (y0, y1) in WINDOWS.items():
            for p in PCTS:
                try:
                    fut_raw = mean_annual_exceedance_days_from_threshold(da_raw, thr[p], y0, y1)
                    fut_qdm = mean_annual_exceedance_days_from_threshold(da_qdm, thr[p], y0, y1)

                    braw = base_raw[p]
                    bqdm = base_qdm[p]

                    rows.append({
                        "model": model,
                        "ssp": ssp,
                        "window": wlab,
                        "percentile": p,
                        "baseline_mean_days_RAW": braw,
                        "future_mean_days_RAW": fut_raw,
                        "delta_days_RAW": fut_raw - braw,
                        "baseline_mean_days_QDM": bqdm,
                        "future_mean_days_QDM": fut_qdm,
                        "delta_days_QDM": fut_qdm - bqdm,
                        "delta_diff_QDM_minus_RAW": (fut_qdm - bqdm) - (fut_raw - braw),
                    })
                except Exception as e:
                    miss.append([model, ssp, f"error window={wlab} p{p}: {e}"])

df = pd.DataFrame(rows)
df.to_csv(OUT_BYMODEL, index=False)

miss_df = pd.DataFrame(miss, columns=["model", "ssp_or_hist", "issue"])
miss_df.to_csv(OUT_MISS, index=False)

# -------------------------
# Ensemble summaries + rank diagnostics (like Step2)
# -------------------------
ens_rows = []
rank_rows = []

if len(df) > 0:
    for p in PCTS:
        dp = df[df["percentile"] == p].copy()
        for ssp in SSPS:
            for wlab in WINDOWS.keys():
                sub = dp[(dp["ssp"] == ssp) & (dp["window"] == wlab)].copy()
                if len(sub) == 0:
                    continue

                raw = sub["delta_days_RAW"].to_numpy(dtype=float)
                qdm = sub["delta_days_QDM"].to_numpy(dtype=float)

                ens_rows.append({
                    "percentile": p,
                    "ssp": ssp,
                    "window": wlab,
                    "n_models": int(sub["model"].nunique()),
                    "mean_delta_days_RAW": float(np.nanmean(raw)),
                    "mean_delta_days_QDM": float(np.nanmean(qdm)),
                    "iqr_delta_days_RAW": iqr(raw),
                    "iqr_delta_days_QDM": iqr(qdm),
                    "mean_delta_diff_QDM_minus_RAW": float(np.nanmean(qdm - raw)),
                    "iqr_ratio_QDM_over_RAW": (iqr(qdm) / iqr(raw)) if iqr(raw) != 0 else np.nan,
                })

                # ranking diagnostics
                rho = spearman(raw, qdm)

                r_raw = pd.Series(raw, index=sub["model"]).rank(ascending=False, method="average")
                r_qdm = pd.Series(qdm, index=sub["model"]).rank(ascending=False, method="average")
                abs_shift = (r_qdm - r_raw).abs()

                top3_raw = set(r_raw.nsmallest(3).index)
                top3_qdm = set(r_qdm.nsmallest(3).index)
                omega3 = len(top3_raw.intersection(top3_qdm)) / 3.0

                rank_rows.append({
                    "percentile": p,
                    "ssp": ssp,
                    "window": wlab,
                    "spearman_rho_RAW_vs_QDM": rho,
                    "mean_abs_rank_shift": float(abs_shift.mean()),
                    "max_abs_rank_shift": float(abs_shift.max()),
                    "top3_overlap_Omega3": float(omega3),
                })

ens_df = pd.DataFrame(ens_rows)
ens_df.to_csv(OUT_ENS, index=False)

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(OUT_RANK, index=False)

print("=== STEP 3 COMPLETE ===")
print("By-model:", OUT_BYMODEL)
print("Ensemble:", OUT_ENS)
print("Rank/IQR:", OUT_RANK)
print("Missing:", OUT_MISS)

if not miss_df.empty:
    print("\nMissing/errors (first 20):")
    print(miss_df.head(20).to_string(index=False))