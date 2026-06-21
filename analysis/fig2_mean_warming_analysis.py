import os
import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# STEP 1 (LOCKED): Mean warming preservation (RAW vs QDM)
# - Paired RAW vs QDM (same model set per SSP/window)
# - Land-only (NaNs ignored)
# - Cosine-lat area weighting (mandatory)
# ============================================================

BASE = r"G:\Paper2_updated\LAND_MASKED"

RAW_HIST_DIR = os.path.join(BASE, "RAW", "HIST")
RAW_SSP_DIRS = {
    "ssp126": os.path.join(BASE, "RAW", "SSP126"),
    "ssp245": os.path.join(BASE, "RAW", "SSP245"),
    "ssp585": os.path.join(BASE, "RAW", "SSP585"),
}

# QDM HIST files are in TWO possible places in your tree:
QDM_HIST_DIR_PRIMARY = os.path.join(BASE, "BC", "HIST")  # preferred
QDM_HIST_DIR_FALLBACK = os.path.join(BASE, "BC")         # fallback (some files here)

QDM_SSP_DIRS = {
    "ssp126": os.path.join(BASE, "BC", "SSP126"),
    "ssp245": os.path.join(BASE, "BC", "SSP245"),
    "ssp585": os.path.join(BASE, "BC", "SSP585"),
}

OUTDIR = os.path.join(BASE, "ANALYSIS_STEP1")
os.makedirs(OUTDIR, exist_ok=True)

MODELS = [
    "BCC-CSM2-MR","CESM2","CMCC-ESM2","CNRM-CM6-1",
    "CNRM-ESM2-1","GFDL-ESM4","MIROC6","MRI-ESM2-0","NorESM2-MM",
]
SSPS = ["ssp126","ssp245","ssp585"]

BASELINE = (1985, 2014)
FUTURE_WINDOWS = {
    "2031-2060": (2031, 2060),
    "2041-2070": (2041, 2070),
    "2071-2100": (2071, 2100),
}

# =========================
# Robust coord handling
# =========================
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
        if t in ds.variables:
            return t
    for name, v in ds.variables.items():
        if str(v.attrs.get("standard_name","")).lower() == "time":
            return name
    raise ValueError("Could not detect time variable.")

def _pick_data_var(ds: xr.Dataset) -> str:
    for v in ["tas","t2m"]:
        if v in ds.data_vars:
            return v
    return list(ds.data_vars.keys())[0]

def open_tas(path: str) -> xr.DataArray:
    ds = xr.open_dataset(path, decode_times=True)
    vname = _pick_data_var(ds)
    da = ds[vname]

    latn = _find_coord_var(ds, "lat")
    lonn = _find_coord_var(ds, "lon")
    timen = _find_time_var(ds)

    if latn not in da.coords: da = da.assign_coords({latn: ds[latn]})
    if lonn not in da.coords: da = da.assign_coords({lonn: ds[lonn]})
    if timen not in da.coords: da = da.assign_coords({timen: ds[timen]})

    rename = {}
    if latn != "latitude": rename[latn] = "latitude"
    if lonn != "longitude": rename[lonn] = "longitude"
    if timen != "time": rename[timen] = "time"
    if rename: da = da.rename(rename)

    return da

def area_weighted_mean_over_land(da: xr.DataArray) -> xr.DataArray:
    lat = da["latitude"]
    w = np.cos(np.deg2rad(lat)).broadcast_like(da)
    valid = da.notnull()
    num = (da * w).where(valid).sum(dim=("latitude","longitude"), skipna=True)
    den = w.where(valid).sum(dim=("latitude","longitude"), skipna=True)
    return num / den

def mean_over_years(ts: xr.DataArray, y0: int, y1: int) -> float:
    years = ts["time"].dt.year
    sub = ts.where((years >= y0) & (years <= y1), drop=True)
    if sub.sizes.get("time", 0) == 0:
        raise ValueError(f"No data in {y0}-{y1}")
    return float(sub.mean(dim="time", skipna=True).values)

# =========================
# File paths
# =========================
def raw_hist_file(model: str) -> str:
    return os.path.join(RAW_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND.nc")

def qdm_hist_file(model: str) -> str:
    fn = f"{model}_HIST_1985_2014_SA_LAND_QDM.nc"
    p1 = os.path.join(QDM_HIST_DIR_PRIMARY, fn)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(QDM_HIST_DIR_FALLBACK, fn)
    return p2  # may or may not exist

def raw_ssp_file(model: str, ssp: str) -> str:
    return os.path.join(RAW_SSP_DIRS[ssp], f"{model}_{ssp.upper()}_2015_2100_SA_LAND.nc")

def qdm_ssp_file(model: str, ssp: str) -> str:
    return os.path.join(QDM_SSP_DIRS[ssp], f"{model}_{ssp.upper()}_2015_2100_SA_LAND_QDM.nc")

# =========================
# Main
# =========================
rows = []
missing = []

for model in MODELS:
    rh = raw_hist_file(model)
    qh = qdm_hist_file(model)

    if not os.path.isfile(rh):
        missing.append((model, "hist", f"missing RAW HIST: {rh}"))
        continue
    if not os.path.isfile(qh):
        missing.append((model, "hist", f"missing QDM HIST: {qh}"))
        continue

    try:
        base_raw_ts = area_weighted_mean_over_land(open_tas(rh))
        base_qdm_ts = area_weighted_mean_over_land(open_tas(qh))
        base_raw = mean_over_years(base_raw_ts, *BASELINE)
        base_qdm = mean_over_years(base_qdm_ts, *BASELINE)
    except Exception as e:
        missing.append((model, "hist", f"error processing HIST: {e}"))
        continue

    for ssp in SSPS:
        rf = raw_ssp_file(model, ssp)
        qf = qdm_ssp_file(model, ssp)

        if not os.path.isfile(rf):
            missing.append((model, ssp, f"missing RAW SSP: {rf}"))
            continue
        if not os.path.isfile(qf):
            missing.append((model, ssp, f"missing QDM SSP: {qf}"))
            continue

        try:
            fut_raw_ts = area_weighted_mean_over_land(open_tas(rf))
            fut_qdm_ts = area_weighted_mean_over_land(open_tas(qf))

            for win, (y0, y1) in FUTURE_WINDOWS.items():
                fut_raw = mean_over_years(fut_raw_ts, y0, y1)
                fut_qdm = mean_over_years(fut_qdm_ts, y0, y1)

                d_raw = fut_raw - base_raw
                d_qdm = fut_qdm - base_qdm
                d_diff = d_qdm - d_raw

                rows.append({
                    "model": model,
                    "ssp": ssp,
                    "window": win,
                    "delta_tas_RAW": d_raw,
                    "delta_tas_QDM": d_qdm,
                    "delta_preservation_diff": d_diff
                })

        except Exception as e:
            missing.append((model, ssp, f"error processing SSP: {e}"))
            continue

df = pd.DataFrame(rows)
missing_df = pd.DataFrame(missing, columns=["model","ssp_or_hist","issue"])

def iqr(x):
    return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

summary = (
    df.groupby(["ssp","window"])
      .agg(
          n_models=("model","nunique"),
          mean_delta_RAW=("delta_tas_RAW","mean"),
          mean_delta_QDM=("delta_tas_QDM","mean"),
          iqr_delta_RAW=("delta_tas_RAW", iqr),
          iqr_delta_QDM=("delta_tas_QDM", iqr),
          mean_preservation_diff=("delta_preservation_diff","mean"),
          iqr_preservation_diff=("delta_preservation_diff", iqr),
      ).reset_index()
)

out_models  = os.path.join(OUTDIR, "step1_delta_tas_by_model.csv")
out_summary = os.path.join(OUTDIR, "step1_delta_tas_ensemble_summary.csv")
out_missing = os.path.join(OUTDIR, "step1_missing_or_errors.csv")

df.to_csv(out_models, index=False)
summary.to_csv(out_summary, index=False)
missing_df.to_csv(out_missing, index=False)

print("=== STEP 1 COMPLETE ===")
print(out_models)
print(out_summary)
print(out_missing)

if not missing_df.empty:
    print("\nMissing/errors (first 20):")
    print(missing_df.head(20).to_string(index=False))