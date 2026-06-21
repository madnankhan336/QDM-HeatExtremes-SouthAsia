#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HISTORICAL-EXTREME VALIDATION (2005-2014 split-sample)
=======================================================

Purpose
-------
Address WACE reviewer concern: validate that QDM not only reproduces the
overall temperature distribution but specifically the EXTREME METRICS
used in this paper, over a held-out historical period that is INDEPENDENT
of the 1985-2004 QDM calibration window.

Design
------
- Calibration window for QDM was 1985-2004.
- Independent split-sample evaluation window: 2005-2014 (10 years).
- For each of 9 CMIP6 models, compute the 6 extreme metrics over 2005-2014:
    1. Annual days above 35 C (regional, land-only, area-weighted)
    2. Annual days above 40 C
    3. Annual P95 exceedance days (P95 from ERA5, 1985-2014 baseline)
    4. Annual P99 exceedance days (P99 from ERA5, 1985-2014 baseline)
    5. HW95 total heatwave days  (>=5 consecutive days above P95)
    6. HW40 mean event duration   (>=3 consecutive days above 40 C)
- Compute biases:
    RAW_bias = RAW(2005-2014) - ERA5(2005-2014)
    QDM_bias = QDM(2005-2014) - ERA5(2005-2014)
- Summarise ensemble:
    mean bias, IQR across models, |bias| improvement (RAW vs QDM)

Outputs
-------
ANALYSIS_VALIDATION/
    TableS7_validation_by_model.csv      (per-model RAW/QDM/ERA5 + biases)
    TableS7_validation_ensemble.csv      (ensemble summary -> goes in paper)
    FigureS2_bias_comparison.png/pdf     (RAW vs QDM bias for 6 metrics)
    validation_missing_or_errors.csv     (any file or computation issues)

Re-uses helpers consistent with Step 1/2/6 conventions.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# =========================================================
# PATHS - matches your existing tree
# =========================================================
BASE = r"G:\Paper2_updated\LAND_MASKED"

RAW_HIST_DIR = os.path.join(BASE, "RAW", "HIST")
QDM_HIST_DIR = os.path.join(BASE, "BC",  "HIST")
ERA5_FILE    = os.path.join(BASE, "ERA5",
                            "ERA5_t2m_AOI_1985_2014_noleap_SA_LAND.nc")

OUTDIR = os.path.join(BASE, "ANALYSIS_VALIDATION")
os.makedirs(OUTDIR, exist_ok=True)

OUT_BYMODEL = os.path.join(OUTDIR, "TableS7_validation_by_model.csv")
OUT_ENS     = os.path.join(OUTDIR, "TableS7_validation_ensemble.csv")
OUT_FIG_PNG = os.path.join(OUTDIR, "FigureS2_bias_comparison.png")
OUT_FIG_PDF = os.path.join(OUTDIR, "FigureS2_bias_comparison.pdf")
OUT_MISS    = os.path.join(OUTDIR, "validation_missing_or_errors.csv")

MODELS = [
    "BCC-CSM2-MR", "CESM2", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1",
    "GFDL-ESM4", "MIROC6", "MRI-ESM2-0", "NorESM2-MM",
]

EVAL_WIN   = (2005, 2014)   # independent split-sample window
PCTL_BASE  = (1985, 2014)   # window for ERA5 P95/P99 thresholds
THRESH_C   = [35.0, 40.0]
HW40_MINLEN = 3
HW95_MINLEN = 5

# =========================================================
# HELPERS (same conventions as Step 1/2/6)
# =========================================================
def _find_coord_var(ds, kind):
    target_std   = "latitude" if kind == "lat" else "longitude"
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
        if kind == "lat" and "lat" in low: cand.append(name)
        if kind == "lon" and "lon" in low: cand.append(name)
    if cand:
        pref = ["lat","latitude"] if kind == "lat" else ["lon","longitude"]
        for p in pref:
            for c in cand:
                if c.lower() == p: return c
        return cand[0]
    raise ValueError(f"Could not detect {kind} variable.")

def _find_time_var(ds):
    for t in ["time", "valid_time"]:
        if t in ds.variables or t in ds.coords:
            return t
    for name, v in ds.variables.items():
        if str(v.attrs.get("standard_name", "")).lower() == "time":
            return name
    raise ValueError("Could not detect time variable.")

def _pick_var(ds):
    for v in ["tas", "t2m", "T2M", "air_temperature"]:
        if v in ds.data_vars:
            return v
    tname = _find_time_var(ds)
    for v in list(ds.data_vars):
        if tname in ds[v].dims:
            return v
    raise ValueError("Could not identify temperature variable (tas/t2m).")

def to_celsius(da):
    units = (da.attrs.get("units", "") or "").lower()
    if units in ["k", "kelvin"]:
        return da - 273.15
    try:
        sample = float(da.isel({da.dims[0]: 0}).mean(skipna=True).values)
        if sample > 100:  # very likely Kelvin
            return da - 273.15
    except Exception:
        pass
    return da

def open_tas(path):
    """Open NetCDF -> DataArray with standard 'time','latitude','longitude' names, in degC."""
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(path, decode_times=time_coder)
    vname = _pick_var(ds)
    da = ds[vname]

    latn = _find_coord_var(ds, "lat")
    lonn = _find_coord_var(ds, "lon")
    timen = _find_time_var(ds)

    if latn  not in da.coords: da = da.assign_coords({latn: ds[latn]})
    if lonn  not in da.coords: da = da.assign_coords({lonn: ds[lonn]})
    if timen not in da.coords: da = da.assign_coords({timen: ds[timen]})

    rename = {}
    if latn  != "latitude":  rename[latn]  = "latitude"
    if lonn  != "longitude": rename[lonn]  = "longitude"
    if timen != "time":      rename[timen] = "time"
    if rename: da = da.rename(rename)

    return to_celsius(da)

def year_select(da, y0, y1):
    yrs = da["time"].dt.year
    return da.where((yrs >= y0) & (yrs <= y1), drop=True)

def area_weighted_mean_over_land(da):
    """Cosine-latitude area-weighted mean across land cells (NaNs ignored)."""
    lat = da["latitude"]
    w   = np.cos(np.deg2rad(lat)).broadcast_like(da)
    valid = da.notnull()
    num = (da * w).where(valid).sum(dim=("latitude", "longitude"), skipna=True)
    den = w.where(valid).sum(dim=("latitude", "longitude"), skipna=True)
    return num / den

# ---------- metric calculators ----------

def annual_exceedance_days_fixed(da, threshold_c, y0, y1):
    """
    Days/yr above absolute threshold (regional area-weighted mean of indicator,
    summed over year).
    """
    sub = year_select(da, y0, y1)
    exc = (sub > threshold_c).astype("float32")
    reg = area_weighted_mean_over_land(exc)   # daily series of areal fraction
    annual = reg.groupby("time.year").sum("time")
    return float(annual.mean().values)

def annual_exceedance_days_pctl_grid(da, pctl_field, y0, y1):
    """
    Days/yr above ERA5-based percentile threshold defined per grid cell.
    pctl_field: 2D (lat,lon) DataArray with threshold per cell.
    """
    sub = year_select(da, y0, y1)
    exc = (sub > pctl_field).astype("float32")
    reg = area_weighted_mean_over_land(exc)
    annual = reg.groupby("time.year").sum("time")
    return float(annual.mean().values)

def heatwave_metrics_fixed(da, threshold_c, y0, y1, min_len):
    """
    Heatwaves defined as >= min_len consecutive days above absolute threshold,
    detected on the regional area-weighted mean series.
    Returns dict with mean event duration and total heatwave days per year.
    Conservative: events must be fully contained in the evaluation window.
    """
    sub = year_select(da, y0, y1)
    reg = area_weighted_mean_over_land(sub)
    vals = reg.values
    flag = vals > threshold_c

    durations = []
    cur = 0
    for f in flag:
        if f:
            cur += 1
        else:
            if cur >= min_len:
                durations.append(cur)
            cur = 0
    if cur >= min_len:
        durations.append(cur)

    n_years = (y1 - y0 + 1)
    total_hw_days = float(sum(durations)) / n_years if n_years > 0 else np.nan
    mean_duration = float(np.mean(durations)) if durations else 0.0
    return {"mean_duration": mean_duration, "total_hw_days_per_year": total_hw_days}

def heatwave_metrics_pctl(da, pctl_field, y0, y1, min_len):
    """
    Heatwaves >= min_len consecutive days above per-grid percentile threshold,
    detected on the regional area-weighted mean of the threshold-exceedance indicator
    converted back to a daily 'regional heatwave-day' definition: a day counts as a
    heatwave day if the area-weighted fraction of cells in exceedance >= 0.5.
    This avoids ambiguity when running consecutive-day logic on a regional series.
    """
    sub = year_select(da, y0, y1)
    exc = (sub > pctl_field).astype("float32")
    reg_frac = area_weighted_mean_over_land(exc)  # 0..1 daily series
    flag = (reg_frac.values >= 0.5)

    durations = []
    cur = 0
    for f in flag:
        if f:
            cur += 1
        else:
            if cur >= min_len:
                durations.append(cur)
            cur = 0
    if cur >= min_len:
        durations.append(cur)

    n_years = (y1 - y0 + 1)
    total_hw_days = float(sum(durations)) / n_years if n_years > 0 else np.nan
    return {"total_hw_days_per_year": total_hw_days}

# =========================================================
# 1) Build ERA5-based per-grid P95/P99 thresholds (1985-2014)
# =========================================================
print("[1] Building ERA5 P95/P99 thresholds over 1985-2014...")
era5 = open_tas(ERA5_FILE)
era5_base = year_select(era5, PCTL_BASE[0], PCTL_BASE[1])

# percentiles per grid cell across time
p95_field = era5_base.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
p99_field = era5_base.quantile(0.99, dim="time", skipna=True).drop_vars("quantile")
print("    P95/P99 fields built.")

# =========================================================
# 2) Compute ERA5 metrics over evaluation window (reference)
# =========================================================
print(f"[2] Computing ERA5 reference metrics over {EVAL_WIN[0]}-{EVAL_WIN[1]}...")
ref = {}
ref["days_above_35C"] = annual_exceedance_days_fixed(era5, 35.0, *EVAL_WIN)
ref["days_above_40C"] = annual_exceedance_days_fixed(era5, 40.0, *EVAL_WIN)
ref["days_above_P95"] = annual_exceedance_days_pctl_grid(era5, p95_field, *EVAL_WIN)
ref["days_above_P99"] = annual_exceedance_days_pctl_grid(era5, p99_field, *EVAL_WIN)

hw40_ref = heatwave_metrics_fixed(era5, 40.0, *EVAL_WIN, HW40_MINLEN)
ref["HW40_mean_duration"] = hw40_ref["mean_duration"]

hw95_ref = heatwave_metrics_pctl(era5, p95_field, *EVAL_WIN, HW95_MINLEN)
ref["HW95_total_days_per_year"] = hw95_ref["total_hw_days_per_year"]

print("    ERA5 reference metrics:")
for k, v in ref.items():
    print(f"      {k:<28s} = {v:.3f}")

# =========================================================
# 3) Loop over models: RAW + QDM in 2005-2014
# =========================================================
print(f"[3] Looping over {len(MODELS)} models for RAW and QDM...")
rows = []
miss = []

for model in MODELS:
    p_raw = os.path.join(RAW_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND.nc")
    p_qdm = os.path.join(QDM_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND_QDM.nc")

    for label, path in [("RAW", p_raw), ("QDM", p_qdm)]:
        if not os.path.isfile(path):
            miss.append([model, label, f"missing file: {path}"])
            continue
        try:
            da = open_tas(path)

            d35 = annual_exceedance_days_fixed(da, 35.0, *EVAL_WIN)
            d40 = annual_exceedance_days_fixed(da, 40.0, *EVAL_WIN)
            dP95 = annual_exceedance_days_pctl_grid(da, p95_field, *EVAL_WIN)
            dP99 = annual_exceedance_days_pctl_grid(da, p99_field, *EVAL_WIN)
            hw40 = heatwave_metrics_fixed(da, 40.0, *EVAL_WIN, HW40_MINLEN)
            hw95 = heatwave_metrics_pctl(da, p95_field, *EVAL_WIN, HW95_MINLEN)

            rows.append({
                "model": model,
                "stream": label,
                "days_above_35C": d35,
                "days_above_40C": d40,
                "days_above_P95": dP95,
                "days_above_P99": dP99,
                "HW40_mean_duration": hw40["mean_duration"],
                "HW95_total_days_per_year": hw95["total_hw_days_per_year"],
            })
            print(f"   [OK] {model} {label}: d35={d35:.2f} d40={d40:.2f} "
                  f"P95={dP95:.2f} P99={dP99:.2f} "
                  f"HW40dur={hw40['mean_duration']:.2f} HW95days={hw95['total_hw_days_per_year']:.2f}")

        except Exception as e:
            miss.append([model, label, f"error: {e}"])
            print(f"   [ERR] {model} {label}: {e}")

df = pd.DataFrame(rows)

# =========================================================
# 4) Compute biases vs ERA5
# =========================================================
metrics = list(ref.keys())
for m in metrics:
    df[f"{m}_bias"] = df[m] - ref[m]

df.to_csv(OUT_BYMODEL, index=False)
print(f"[4] Wrote per-model table: {OUT_BYMODEL}")

# =========================================================
# 5) Ensemble summary table (for the paper)
# =========================================================
def iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

ens_rows = []
for stream in ["RAW", "QDM"]:
    sub = df[df["stream"] == stream]
    if sub.empty:
        continue
    row = {"stream": stream, "n_models": int(sub["model"].nunique())}
    for m in metrics:
        b = sub[f"{m}_bias"].to_numpy(dtype=float)
        row[f"{m}_ERA5"]       = float(ref[m])
        row[f"{m}_mean"]       = float(np.nanmean(sub[m]))
        row[f"{m}_mean_bias"]  = float(np.nanmean(b))
        row[f"{m}_iqr_bias"]   = iqr(b)
        row[f"{m}_mean_absbias"] = float(np.nanmean(np.abs(b)))
    ens_rows.append(row)

ens = pd.DataFrame(ens_rows)

# Add a row computing the |bias| reduction QDM vs RAW (percent improvement)
if {"RAW", "QDM"}.issubset(set(ens["stream"])):
    raw_row = ens[ens["stream"] == "RAW"].iloc[0]
    qdm_row = ens[ens["stream"] == "QDM"].iloc[0]
    impr = {"stream": "abs_bias_reduction_pct"}
    for m in metrics:
        a = raw_row[f"{m}_mean_absbias"]
        b = qdm_row[f"{m}_mean_absbias"]
        impr[f"{m}_ERA5"]         = ""
        impr[f"{m}_mean"]         = ""
        impr[f"{m}_mean_bias"]    = ""
        impr[f"{m}_iqr_bias"]     = ""
        impr[f"{m}_mean_absbias"] = (100.0 * (a - b) / a) if a not in (0, np.nan) else np.nan
    ens = pd.concat([ens, pd.DataFrame([impr])], ignore_index=True)

ens.to_csv(OUT_ENS, index=False)
print(f"[5] Wrote ensemble summary: {OUT_ENS}")

# =========================================================
# 6) Figure S2 - bias comparison plot
# =========================================================
print("[6] Drawing Figure S2 (bias comparison)...")
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
metric_labels = {
    "days_above_35C":        "Days > 35 \u00b0C (per yr)",
    "days_above_40C":        "Days > 40 \u00b0C (per yr)",
    "days_above_P95":        "Days > P95 (per yr)",
    "days_above_P99":        "Days > P99 (per yr)",
    "HW40_mean_duration":    "HW40 mean duration (days)",
    "HW95_total_days_per_year": "HW95 days (per yr)",
}

for ax, m in zip(axes.flatten(), metrics):
    raw_b = df.loc[df["stream"] == "RAW", f"{m}_bias"].to_numpy(dtype=float)
    qdm_b = df.loc[df["stream"] == "QDM", f"{m}_bias"].to_numpy(dtype=float)
    bp = ax.boxplot([raw_b, qdm_b], labels=["RAW", "QDM"], widths=0.55,
                    patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], ["#d9d9d9", "#9ecae1"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title(metric_labels.get(m, m), fontsize=10.5)
    ax.set_ylabel("Bias vs ERA5", fontsize=9.5)
    ax.tick_params(labelsize=9)

fig.suptitle("Historical-extreme bias validation against ERA5 "
             f"({EVAL_WIN[0]}\u2013{EVAL_WIN[1]} split-sample window)",
             fontsize=12, fontweight="bold", y=1.00)
plt.tight_layout()
fig.savefig(OUT_FIG_PNG, dpi=400, bbox_inches="tight")
fig.savefig(OUT_FIG_PDF,            bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {OUT_FIG_PNG}")
print(f"    Saved: {OUT_FIG_PDF}")

# =========================================================
# 7) Missing log
# =========================================================
miss_df = pd.DataFrame(miss, columns=["model", "stream", "issue"])
miss_df.to_csv(OUT_MISS, index=False)
if not miss_df.empty:
    print("\n[WARN] Some files/computations failed - see:", OUT_MISS)
    print(miss_df.head(20).to_string(index=False))

print("\n=== VALIDATION COMPLETE ===")
print("Use TableS7_validation_ensemble.csv to fill in the paper text.")