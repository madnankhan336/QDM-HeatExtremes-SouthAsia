#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 6 — Quantile-dependent uncertainty transformation (LOCKED DESIGN)

For q = 0.50, 0.90, 0.95, 0.99:
- per model: compute regional daily quantile of temperature (RAW and QDM)
- across models: compute IQR_RAW(q), IQR_QDM(q)
- R_IQR(q) = IQR_QDM(q) / IQR_RAW(q)
- Tail reshaping index: S = R_IQR(0.99) / R_IQR(0.50)

Consistent with Steps 1–5:
- build cos(lat) area-weighted regional mean daily series first (land via NaNs)
- strict pairing RAW vs QDM (same model set per SSP/window)
- baseline 1985–2014, windows: 2031–2060, 2041–2070, 2071–2100

Outputs:
ANALYSIS_STEP6/step6_quantiles_by_model.csv
ANALYSIS_STEP6/step6_uncertainty_by_quantile.csv
ANALYSIS_STEP6/step6_tail_reshaping_index.csv
ANALYSIS_STEP6/step6_missing_or_errors.csv
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

OUTDIR = os.path.join(BASE, "ANALYSIS_STEP6")
os.makedirs(OUTDIR, exist_ok=True)

OUT_BYMODEL = os.path.join(OUTDIR, "step6_quantiles_by_model.csv")
OUT_UNC     = os.path.join(OUTDIR, "step6_uncertainty_by_quantile.csv")
OUT_S       = os.path.join(OUTDIR, "step6_tail_reshaping_index.csv")
OUT_MISS    = os.path.join(OUTDIR, "step6_missing_or_errors.csv")

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

QS = [0.50, 0.90, 0.95, 0.99]

# -------------------------
# Robust helpers (same as Step4/5)
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
# Main compute
# -------------------------
rows = []
miss = []

def quantiles_for_period(ts: xr.DataArray, y0: int, y1: int) -> dict:
    sub = year_select(ts, y0, y1)
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"No data in {y0}-{y1}")
    out = {}
    vals = sub.values.astype(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        for q in QS:
            out[f"q{int(q*100):02d}"] = np.nan
        return out
    for q in QS:
        out[f"q{int(q*100):02d}"] = float(np.nanquantile(vals, q))
    return out

# Baseline quantiles are useful too; we compute baseline + futures
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

    # baseline (1985–2014)
    try:
        q_raw_base = quantiles_for_period(ts_raw_hist, BASELINE[0], BASELINE[1])
        q_qdm_base = quantiles_for_period(ts_qdm_hist, BASELINE[0], BASELINE[1])
        rows.append({"model": model, "ssp": "hist", "window": "1985-2014", "stream": "RAW", **q_raw_base})
        rows.append({"model": model, "ssp": "hist", "window": "1985-2014", "stream": "QDM", **q_qdm_base})
    except Exception as e:
        miss.append([model, "hist", f"error baseline quantiles: {e}"])

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
                q_raw = quantiles_for_period(ts_raw, y0, y1)
                q_qdm = quantiles_for_period(ts_qdm, y0, y1)
                rows.append({"model": model, "ssp": ssp, "window": wlab, "stream": "RAW", **q_raw})
                rows.append({"model": model, "ssp": ssp, "window": wlab, "stream": "QDM", **q_qdm})
            except Exception as e:
                miss.append([model, ssp, f"error window={wlab}: {e}"])

df = pd.DataFrame(rows)
df.to_csv(OUT_BYMODEL, index=False)

miss_df = pd.DataFrame(miss, columns=["model","ssp_or_hist","issue"])
miss_df.to_csv(OUT_MISS, index=False)

# -------------------------
# Across-model uncertainty by quantile (paired RAW vs QDM)
# -------------------------
unc_rows = []
s_rows = []

if len(df) > 0:
    qcols = [f"q{int(q*100):02d}" for q in QS]

    for ssp in ["hist"] + SSPS:
        windows = ["1985-2014"] if ssp == "hist" else list(WINDOWS.keys())
        for wlab in windows:
            sub = df[(df["ssp"] == ssp) & (df["window"] == wlab)].copy()
            if len(sub) == 0:
                continue

            raw = sub[sub["stream"] == "RAW"].set_index("model")
            qdm = sub[sub["stream"] == "QDM"].set_index("model")
            common = raw.index.intersection(qdm.index)
            raw = raw.loc[common]
            qdm = qdm.loc[common]
            if len(common) == 0:
                continue

            r_by_q = {}
            for qc in qcols:
                x = raw[qc].to_numpy(dtype=float)
                y = qdm[qc].to_numpy(dtype=float)
                i_raw = iqr(x)
                i_qdm = iqr(y)
                r = (i_qdm / i_raw) if (np.isfinite(i_raw) and i_raw != 0) else np.nan

                unc_rows.append({
                    "ssp": ssp,
                    "window": wlab,
                    "quantile": qc,
                    "n_models_paired": int(len(common)),
                    "iqr_RAW": float(i_raw),
                    "iqr_QDM": float(i_qdm),
                    "R_IQR_QDM_over_RAW": float(r) if np.isfinite(r) else np.nan,
                    "mean_q_RAW": float(np.nanmean(x)),
                    "mean_q_QDM": float(np.nanmean(y)),
                    "mean_diff_QDM_minus_RAW": float(np.nanmean(y - x)),
                })
                r_by_q[qc] = r

            # Tail reshaping index S = R(0.99)/R(0.50)
            r99 = r_by_q.get("q99", np.nan)
            r50 = r_by_q.get("q50", np.nan)
            S = (r99 / r50) if (np.isfinite(r99) and np.isfinite(r50) and r50 != 0) else np.nan
            s_rows.append({
                "ssp": ssp,
                "window": wlab,
                "R_IQR_q50": r50,
                "R_IQR_q99": r99,
                "Tail_reshaping_index_S": S,
            })

unc_df = pd.DataFrame(unc_rows)
unc_df.to_csv(OUT_UNC, index=False)

s_df = pd.DataFrame(s_rows)
s_df.to_csv(OUT_S, index=False)

print("=== STEP 6 COMPLETE ===")
print("By-model:", OUT_BYMODEL)
print("Uncertainty:", OUT_UNC)
print("Tail index S:", OUT_S)
print("Missing:", OUT_MISS)

if not miss_df.empty:
    print("\nMissing/errors (first 20):")
    print(miss_df.head(20).to_string(index=False))