#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 7 — Decoupling metric: tail amplification per °C (LOCKED DESIGN)

A = (Δ hazard) / (Δ mean warming)

Inputs:
- Step 1 warming deltas:
  ANALYSIS_STEP1/step1_delta_tas_by_model.csv
- Step 2 absolute exceedance deltas:
  ANALYSIS_STEP2/step2_exceedance_by_model.csv
- Step 3 percentile exceedance deltas:
  ANALYSIS_STEP3/step3_percentile_exceedance_by_model.csv
- Step 4 heatwave deltas:
  ANALYSIS_STEP4/step4_heatwave_by_model.csv

Outputs:
ANALYSIS_STEP7/step7_A_by_model.csv
ANALYSIS_STEP7/step7_A_ensemble_summary.csv
ANALYSIS_STEP7/step7_A_rank_and_uncertainty.csv
ANALYSIS_STEP7/step7_missing_or_errors.csv

Notes:
- Strict pairing: model must exist in warming + hazard tables
- Avoid divide-by-zero: if ΔT is 0 or NaN => A is NaN
"""

import os
import numpy as np
import pandas as pd

BASE = r"G:\Paper2_updated\LAND_MASKED"

IN_STEP1 = os.path.join(BASE, "ANALYSIS_STEP1", "step1_delta_tas_by_model.csv")
IN_STEP2 = os.path.join(BASE, "ANALYSIS_STEP2", "step2_exceedance_by_model.csv")
IN_STEP3 = os.path.join(BASE, "ANALYSIS_STEP3", "step3_percentile_exceedance_by_model.csv")
IN_STEP4 = os.path.join(BASE, "ANALYSIS_STEP4", "step4_heatwave_by_model.csv")

OUTDIR = os.path.join(BASE, "ANALYSIS_STEP7")
os.makedirs(OUTDIR, exist_ok=True)

OUT_BYMODEL = os.path.join(OUTDIR, "step7_A_by_model.csv")
OUT_ENS     = os.path.join(OUTDIR, "step7_A_ensemble_summary.csv")
OUT_RANK    = os.path.join(OUTDIR, "step7_A_rank_and_uncertainty.csv")
OUT_MISS    = os.path.join(OUTDIR, "step7_missing_or_errors.csv")

SSPS = ["ssp126", "ssp245", "ssp585"]
WINDOWS = ["2031-2060", "2041-2070", "2071-2100"]

def safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den
    out[(~np.isfinite(out)) | (den == 0)] = np.nan
    return out

def iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

miss = []

# -------------------------
# Load Step 1 warming
# -------------------------
if not os.path.isfile(IN_STEP1):
    raise FileNotFoundError(f"Missing Step1 file: {IN_STEP1}")

w = pd.read_csv(IN_STEP1)
# expected columns: model, ssp, window, delta_tas_RAW, delta_tas_QDM
req = {"model","ssp","window","delta_tas_RAW","delta_tas_QDM"}
if not req.issubset(set(w.columns)):
    raise ValueError(f"Step1 file missing columns. Need {req}, got {set(w.columns)}")

w = w[w["ssp"].isin(SSPS) & w["window"].isin(WINDOWS)].copy()

# -------------------------
# Helper to compute A from a hazard table
# -------------------------
def compute_A_from_hazard(df_haz, haz_name, key_cols, delta_raw_col, delta_qdm_col, extra_cols=None):
    """
    Returns standardized A rows with columns:
    model, ssp, window, hazard_type, hazard_id, A_RAW, A_QDM, A_diff(QDM-RAW), plus any extra cols.
    """
    if extra_cols is None:
        extra_cols = []

    # merge with warming on model/ssp/window
    m = df_haz.merge(w, on=["model","ssp","window"], how="inner")

    if m.empty:
        miss.append([haz_name, "merge", "No overlapping rows after merging with Step1 warming"])
        return pd.DataFrame()

    m["A_RAW"] = safe_div(m[delta_raw_col], m["delta_tas_RAW"])
    m["A_QDM"] = safe_div(m[delta_qdm_col], m["delta_tas_QDM"])
    m["A_diff_QDM_minus_RAW"] = m["A_QDM"] - m["A_RAW"]

    keep = ["model","ssp","window"] + extra_cols + ["A_RAW","A_QDM","A_diff_QDM_minus_RAW"]
    out = m[keep].copy()

    # add hazard identity columns
    out.insert(3, "hazard_type", haz_name)
    # build hazard_id from key_cols
    if key_cols:
        out.insert(4, "hazard_id", out[key_cols].astype(str).agg("_".join, axis=1))
    else:
        out.insert(4, "hazard_id", haz_name)

    return out

# -------------------------
# Step 2 hazards (absolute exceedance)
# -------------------------
A_parts = []

if os.path.isfile(IN_STEP2):
    s2 = pd.read_csv(IN_STEP2)
    # expected: model, ssp, window, threshold_C, delta_days_RAW, delta_days_QDM
    need = {"model","ssp","window","threshold_C","delta_days_RAW","delta_days_QDM"}
    if need.issubset(set(s2.columns)):
        s2 = s2[s2["ssp"].isin(SSPS) & s2["window"].isin(WINDOWS)].copy()
        A2 = compute_A_from_hazard(
            s2,
            haz_name="exceed_abs",
            key_cols=["threshold_C"],
            delta_raw_col="delta_days_RAW",
            delta_qdm_col="delta_days_QDM",
            extra_cols=["threshold_C"]
        )
        A_parts.append(A2)
    else:
        miss.append(["step2", "columns", f"Missing required columns: {need - set(s2.columns)}"])
else:
    miss.append(["step2", "missing", f"File not found: {IN_STEP2}"])

# -------------------------
# Step 3 hazards (percentile exceedance)
# -------------------------
if os.path.isfile(IN_STEP3):
    s3 = pd.read_csv(IN_STEP3)
    # expected: model, ssp, window, percentile, delta_days_RAW, delta_days_QDM
    need = {"model","ssp","window","percentile","delta_days_RAW","delta_days_QDM"}
    if need.issubset(set(s3.columns)):
        s3 = s3[s3["ssp"].isin(SSPS) & s3["window"].isin(WINDOWS)].copy()
        A3 = compute_A_from_hazard(
            s3,
            haz_name="exceed_pct",
            key_cols=["percentile"],
            delta_raw_col="delta_days_RAW",
            delta_qdm_col="delta_days_QDM",
            extra_cols=["percentile"]
        )
        A_parts.append(A3)
    else:
        miss.append(["step3", "columns", f"Missing required columns: {need - set(s3.columns)}"])
else:
    miss.append(["step3", "missing", f"File not found: {IN_STEP3}"])

# -------------------------
# Step 4 hazards (heatwave metrics)
# -------------------------
if os.path.isfile(IN_STEP4):
    s4 = pd.read_csv(IN_STEP4)
    # expected: model, ssp, window, heatwave, metric, delta_RAW, delta_QDM
    need = {"model","ssp","window","heatwave","metric","delta_RAW","delta_QDM"}
    if need.issubset(set(s4.columns)):
        s4 = s4[s4["ssp"].isin(SSPS) & s4["window"].isin(WINDOWS)].copy()
        A4 = compute_A_from_hazard(
            s4,
            haz_name="heatwave",
            key_cols=["heatwave","metric"],
            delta_raw_col="delta_RAW",
            delta_qdm_col="delta_QDM",
            extra_cols=["heatwave","metric"]
        )
        A_parts.append(A4)
    else:
        miss.append(["step4", "columns", f"Missing required columns: {need - set(s4.columns)}"])
else:
    miss.append(["step4", "missing", f"File not found: {IN_STEP4}"])

# -------------------------
# Combine + save by-model
# -------------------------
A = pd.concat([p for p in A_parts if p is not None and len(p) > 0], ignore_index=True)

if A.empty:
    raise RuntimeError("No A results produced. Check that Steps 1–4 outputs exist and share model/ssp/window keys.")

A.to_csv(OUT_BYMODEL, index=False)

# -------------------------
# Ensemble summary
# -------------------------
ens_rows = []
for (haz, hid, ssp, win), g in A.groupby(["hazard_type","hazard_id","ssp","window"]):
    x = g["A_RAW"].to_numpy(dtype=float)
    y = g["A_QDM"].to_numpy(dtype=float)
    ens_rows.append({
        "hazard_type": haz,
        "hazard_id": hid,
        "ssp": ssp,
        "window": win,
        "n_models_paired": int(g["model"].nunique()),
        "mean_A_RAW": float(np.nanmean(x)),
        "mean_A_QDM": float(np.nanmean(y)),
        "iqr_A_RAW": iqr(x),
        "iqr_A_QDM": iqr(y),
        "mean_A_diff_QDM_minus_RAW": float(np.nanmean(y - x)),
        "iqr_ratio_QDM_over_RAW": (iqr(y) / iqr(x)) if iqr(x) != 0 else np.nan,
    })

ens = pd.DataFrame(ens_rows)
ens.to_csv(OUT_ENS, index=False)

# -------------------------
# Ranking diagnostics
# -------------------------
rank_rows = []
for (haz, hid, ssp, win), g in A.groupby(["hazard_type","hazard_id","ssp","window"]):
    # ranks based on A (higher A = stronger tail amplification)
    sub = g.set_index("model")
    x = sub["A_RAW"].to_numpy(dtype=float)
    y = sub["A_QDM"].to_numpy(dtype=float)

    rho = spearman(x, y)

    r_raw = pd.Series(x, index=sub.index).rank(ascending=False, method="average")
    r_qdm = pd.Series(y, index=sub.index).rank(ascending=False, method="average")
    abs_shift = (r_qdm - r_raw).abs()

    top3_raw = set(r_raw.nsmallest(3).index)
    top3_qdm = set(r_qdm.nsmallest(3).index)
    omega3 = len(top3_raw.intersection(top3_qdm)) / 3.0

    rank_rows.append({
        "hazard_type": haz,
        "hazard_id": hid,
        "ssp": ssp,
        "window": win,
        "spearman_rho_RAW_vs_QDM": float(rho),
        "mean_abs_rank_shift": float(abs_shift.mean()),
        "max_abs_rank_shift": float(abs_shift.max()),
        "top3_overlap_Omega3": float(omega3),
    })

rank = pd.DataFrame(rank_rows)
rank.to_csv(OUT_RANK, index=False)

# -------------------------
# Missing log
# -------------------------
miss_df = pd.DataFrame(miss, columns=["source","type","issue"])
miss_df.to_csv(OUT_MISS, index=False)

print("=== STEP 7 COMPLETE ===")
print("By-model:", OUT_BYMODEL)
print("Ensemble:", OUT_ENS)
print("Rank/IQR:", OUT_RANK)
print("Missing:", OUT_MISS)

if not miss_df.empty:
    print("\nMissing/errors (first 20):")
    print(miss_df.head(20).to_string(index=False))