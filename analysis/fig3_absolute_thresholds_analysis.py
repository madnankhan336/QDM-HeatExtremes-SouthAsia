# ============================================================
# STEP 2 (Paper 2 locked design) - Absolute hazard exceedances
# Days >35C and Days >40C
#
# Inputs (your structure):
#   RAW/HIST/<MODEL>_HIST_1985_2014_SA_LAND.nc
#   RAW/SSPxxx/<MODEL>_SSPxxx_2015_2100_SA_LAND.nc
#   BC/HIST/<MODEL>_HIST_1985_2014_SA_LAND_QDM.nc
#   BC/SSPxxx/<MODEL>_SSPxxx_2015_2100_SA_LAND_QDM.nc
#
# Outputs:
#   ANALYSIS_STEP2/step2_exceedance_by_model.csv
#   ANALYSIS_STEP2/step2_exceedance_ensemble_summary.csv
#   ANALYSIS_STEP2/step2_rank_and_uncertainty.csv
#   ANALYSIS_STEP2/step2_missing_or_errors.csv
#
# Non-negotiables satisfied:
# - Land-only (already masked files)
# - Cosine-lat area weighting
# - Identical paired model sets (RAW vs QDM)
# - Baseline 1985–2014, futures: 2031–2060, 2041–2070, 2071–2100
# ============================================================

import os
import numpy as np
import pandas as pd
import xarray as xr

# -------------------------
# Paths
# -------------------------
BASE = r"G:\Paper2_updated\LAND_MASKED"
RAW_HIST_DIR = os.path.join(BASE, "RAW", "HIST")
RAW_SSP_DIR  = os.path.join(BASE, "RAW")
QDM_HIST_DIR = os.path.join(BASE, "BC", "HIST")
QDM_SSP_DIR  = os.path.join(BASE, "BC")

OUTDIR = os.path.join(BASE, "ANALYSIS_STEP2")
os.makedirs(OUTDIR, exist_ok=True)

OUT_BYMODEL = os.path.join(OUTDIR, "step2_exceedance_by_model.csv")
OUT_ENS     = os.path.join(OUTDIR, "step2_exceedance_ensemble_summary.csv")
OUT_RANK    = os.path.join(OUTDIR, "step2_rank_and_uncertainty.csv")
OUT_MISS    = os.path.join(OUTDIR, "step2_missing_or_errors.csv")

# Step 1 warming file (for tail amplification per degree warming later)
STEP1_BYMODEL = os.path.join(BASE, "ANALYSIS_STEP1", "step1_delta_tas_by_model.csv")

MODELS = [
    "BCC-CSM2-MR", "CESM2", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1",
    "GFDL-ESM4", "MIROC6", "MRI-ESM2-0", "NorESM2-MM"
]

SSPS = ["ssp126", "ssp245", "ssp585"]
WINDOWS = {
    "2031-2060": (2031, 2060),
    "2041-2070": (2041, 2070),
    "2071-2100": (2071, 2100),
}
BASELINE = (1985, 2014)

THRESH_C = [35.0, 40.0]  # locked primary thresholds

# -------------------------
# Helpers
# -------------------------
def find_varname(ds):
    # prefer tas, then t2m, then any 3D/2D numeric variable with time dim
    for cand in ["tas", "t2m", "T2M", "air_temperature"]:
        if cand in ds.data_vars:
            return cand
    # fallback: first data var that has time dimension
    for v in list(ds.data_vars):
        if "time" in ds[v].dims:
            return v
    raise ValueError("Could not identify temperature variable (tas/t2m).")

def find_latname(ds):
    for cand in ["lat", "latitude", "Latitude", "LAT"]:
        if cand in ds.coords or cand in ds.variables:
            return cand
    # sometimes lat is a coordinate attached to another dim
    for c in ds.coords:
        if "lat" in c.lower():
            return c
    raise ValueError("Could not find latitude coordinate.")

def to_celsius(da):
    units = (da.attrs.get("units", "") or "").lower()
    if "k" in units and "degc" not in units and "c" not in units:
        return da - 273.15
    return da

def year_select(da, y0, y1):
    # works for datetime or cftime; safe via .dt.year
    yrs = da["time"].dt.year
    return da.where((yrs >= y0) & (yrs <= y1), drop=True)

def cosine_weights(ds, lat_name, spatial_dims):
    lat = ds[lat_name]
    # broadcast lat to data dims if needed
    w = np.cos(np.deg2rad(lat))
    # ensure weights align to spatial dims
    if set(w.dims) != set(spatial_dims):
        w = w.broadcast_like(ds[spatial_dims[0]] if spatial_dims[0] in ds.coords else ds[list(ds.data_vars)[0]].isel(time=0))
        # after broadcast, trim to spatial dims if extra
        w = w.rename({d: d for d in w.dims})
    return w

def spatial_dims_of(da):
    return [d for d in da.dims if d != "time"]

def area_weighted_mean(da, w):
    # da: (time, ...spatial...)
    spatial = spatial_dims_of(da)
    # mask any NaN cells (land mask may set sea to NaN)
    valid = xr.where(np.isfinite(da.isel(time=0)), 1.0, np.nan)
    ww = w * valid
    num = (da * ww).sum(dim=spatial, skipna=True)
    den = ww.sum(dim=spatial, skipna=True)
    return num / den

def mean_annual_exceedance_days(ds_path, thresh_c, y0, y1):
    # Open
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(ds_path, decode_times=time_coder)

    vname = find_varname(ds)
    latname = find_latname(ds)
    da = to_celsius(ds[vname])

    # time subset
    da = year_select(da, y0, y1)

    # build weights
    spatial = spatial_dims_of(da)
    w = cosine_weights(ds, latname, spatial)

    # exceedance (time, spatial...) -> bool -> daily count per year
    exceed = (da > thresh_c).astype("float32")
    reg_exceed = area_weighted_mean(exceed, w)  # time series of fraction(>thresh)
    # Since land-only cells are present, this returns fraction; we need "days"
    # Convert to days per year by summing daily 0/1 at each time step (already fraction)
    # Here reg_exceed is fraction of area exceeding each day. For *regional mean exceedance days*
    # we need mean annual count of days where threshold exceeded *somewhere*?
    #
    # Locked design expects area-weighted mean of exceedance indicator across land,
    # summed over days -> "area-weighted exceedance days" (equivalent to mean number of exceedance-days per cell).
    #
    # That is: average over space of exceed(0/1) then sum over time.
    # So reg_exceed already is spatial mean(0/1). Sum over days -> "mean exceedance days".
    years = reg_exceed["time"].dt.year
    annual = reg_exceed.groupby(years).sum("time")  # days/year (area-weighted mean)
    return float(annual.mean().values)  # mean annual exceedance days

# Spearman without scipy (rank then Pearson)
def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0,1])

# -------------------------
# Main compute
# -------------------------
rows = []
miss = []

def path_raw_hist(model):
    return os.path.join(RAW_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND.nc")

def path_qdm_hist(model):
    return os.path.join(QDM_HIST_DIR, f"{model}_HIST_1985_2014_SA_LAND_QDM.nc")

def path_raw_ssp(model, ssp):
    return os.path.join(RAW_SSP_DIR, ssp.upper(), f"{model}_{ssp.upper()}_2015_2100_SA_LAND.nc")

def path_qdm_ssp(model, ssp):
    return os.path.join(QDM_SSP_DIR, ssp.upper(), f"{model}_{ssp.upper()}_2015_2100_SA_LAND_QDM.nc")

for model in MODELS:
    # Baseline files must exist for both RAW and QDM (paired rule)
    p_raw_hist = path_raw_hist(model)
    p_qdm_hist = path_qdm_hist(model)

    if not os.path.isfile(p_raw_hist):
        miss.append([model, "hist", f"missing RAW HIST: {p_raw_hist}"])
        continue
    if not os.path.isfile(p_qdm_hist):
        miss.append([model, "hist", f"missing QDM HIST: {p_qdm_hist}"])
        continue

    # Pre-compute baseline for both streams for each threshold
    base_raw = {}
    base_qdm = {}
    for thr in THRESH_C:
        try:
            base_raw[thr] = mean_annual_exceedance_days(p_raw_hist, thr, BASELINE[0], BASELINE[1])
        except Exception as e:
            miss.append([model, "hist", f"RAW HIST error thr={thr}: {e}"])
            base_raw[thr] = np.nan
        try:
            base_qdm[thr] = mean_annual_exceedance_days(p_qdm_hist, thr, BASELINE[0], BASELINE[1])
        except Exception as e:
            miss.append([model, "hist", f"QDM HIST error thr={thr}: {e}"])
            base_qdm[thr] = np.nan

    for ssp in SSPS:
        p_raw = path_raw_ssp(model, ssp)
        p_qdm = path_qdm_ssp(model, ssp)

        # paired rule: require both raw and qdm scenario file
        if not os.path.isfile(p_raw):
            miss.append([model, ssp, f"missing RAW {ssp}: {p_raw}"])
            continue
        if not os.path.isfile(p_qdm):
            miss.append([model, ssp, f"missing QDM {ssp}: {p_qdm}"])
            continue

        for wlab, (y0, y1) in WINDOWS.items():
            for thr in THRESH_C:
                try:
                    fut_raw = mean_annual_exceedance_days(p_raw, thr, y0, y1)
                    fut_qdm = mean_annual_exceedance_days(p_qdm, thr, y0, y1)

                    braw = base_raw[thr]
                    bqdm = base_qdm[thr]

                    rows.append({
                        "model": model,
                        "ssp": ssp,
                        "window": wlab,
                        "threshold_C": thr,
                        "baseline_mean_days_RAW": braw,
                        "future_mean_days_RAW": fut_raw,
                        "delta_days_RAW": fut_raw - braw,
                        "baseline_mean_days_QDM": bqdm,
                        "future_mean_days_QDM": fut_qdm,
                        "delta_days_QDM": fut_qdm - bqdm,
                        "delta_diff_QDM_minus_RAW": (fut_qdm - bqdm) - (fut_raw - braw),
                    })
                except Exception as e:
                    miss.append([model, ssp, f"error window={wlab} thr={thr}: {e}"])

df = pd.DataFrame(rows)
df.to_csv(OUT_BYMODEL, index=False)

miss_df = pd.DataFrame(miss, columns=["model", "ssp_or_hist", "issue"])
miss_df.to_csv(OUT_MISS, index=False)

# -------------------------
# Ensemble summaries + rank diagnostics
# -------------------------
ens_rows = []
rank_rows = []

if len(df) > 0:
    for thr in THRESH_C:
        dthr = df[df["threshold_C"] == thr].copy()
        for ssp in SSPS:
            for wlab in WINDOWS.keys():
                sub = dthr[(dthr["ssp"] == ssp) & (dthr["window"] == wlab)].copy()
                if len(sub) == 0:
                    continue

                raw = sub["delta_days_RAW"].to_numpy()
                qdm = sub["delta_days_QDM"].to_numpy()

                def iqr(x):
                    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

                ens_rows.append({
                    "threshold_C": thr,
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

                # Ranking diagnostics (required later)
                rho = spearman(raw, qdm)

                # Rank shifts
                r_raw = pd.Series(raw, index=sub["model"]).rank(ascending=False, method="average")
                r_qdm = pd.Series(qdm, index=sub["model"]).rank(ascending=False, method="average")
                abs_shift = (r_qdm - r_raw).abs()

                # top-3 overlap
                top3_raw = set(r_raw.nsmallest(3).index)
                top3_qdm = set(r_qdm.nsmallest(3).index)
                omega3 = len(top3_raw.intersection(top3_qdm)) / 3.0

                rank_rows.append({
                    "threshold_C": thr,
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

print("=== STEP 2 COMPLETE ===")
print("By-model:", OUT_BYMODEL)
print("Ensemble:", OUT_ENS)
print("Rank/IQR:", OUT_RANK)
print("Missing:", OUT_MISS)

# -------------------------
# Optional quick check: add A = (Δ exceedance days)/(Δ mean warming) later
# -------------------------
if os.path.isfile(STEP1_BYMODEL) and len(df) > 0:
    step1 = pd.read_csv(STEP1_BYMODEL)
    # join on model/ssp/window
    merged = df.merge(step1, on=["model", "ssp", "window"], how="left")
    merged["A_RAW"] = merged["delta_days_RAW"] / merged["delta_tas_RAW"]
    merged["A_QDM"] = merged["delta_days_QDM"] / merged["delta_tas_QDM"]
    merged_out = os.path.join(OUTDIR, "step2_tail_amplification_A_by_model.csv")
    merged.to_csv(merged_out, index=False)
    print("Tail amplification (A) by model:", merged_out)