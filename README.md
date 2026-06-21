# Trend-preserving bias correction reshapes future heat-risk metrics and ensemble uncertainty over South Asia

Analysis and figure code Trend-preserving bias correction reshapes future heat-risk metrics and ensemble uncertainty over South Asia.
The workflow compares **RAW** CMIP6 daily-mean near-surface air temperature (`tas`) against **QDM**
(Quantile Delta Mapping) bias-corrected output from nine CMIP6 models over South Asia
(58–98°E, 2–38°N) under SSP1-2.6, SSP2-4.5 and SSP5-8.5, for a 1985–2014 baseline and three
future windows (2031–2060, 2041–2070, 2071–2100).

Code is split into two folders: **`analysis/`** (Python) computes the per-model and ensemble
summary tables; **`figures/`** (R) reads those tables and draws the manuscript figures. The
spatial figure (Fig. 9) and the validation outputs (Table S7, Fig. S2) are produced directly in
Python.

---



## Repository layout

```
heat-risk-qdm-south-asia/
├── README.md
├── LICENSE
├── requirements.txt          # Python dependencies
├── r_packages.txt            # R packages used by figures/
├── .gitignore
├── analysis/                 # Python: computes the CSV summary tables
│   ├── fig2_mean_warming_analysis.py
│   ├── fig3_absolute_thresholds_analysis.py
│   ├── fig4_percentile_exceedances_analysis.py
│   ├── fig5_heatwave_persistence_analysis.py
│   ├── fig6_temporal_dependence_analysis.py
│   ├── fig7_quantile_uncertainty_analysis.py
│   ├── fig8_tail_amplification_analysis.py
│   ├── fig9_spatial_diagnostics.py
│   └── validation_tableS7_figS2.py
└── figures/                  # R: reads the CSV tables and draws the figures
    ├── fig2_mean_warming.R
    ├── fig3_absolute_thresholds.R
    ├── fig4_percentile_exceedances.R
    ├── fig5_and_fig6_heatwave_and_temporal.R
    ├── fig7_quantile_uncertainty.R
    └── fig8_tail_amplification.R
```


## Conventions (identical across all scripts)
- Land-only cells (sea set to NaN and ignored).
- Cosine-latitude area weighting for all regional means.
- Strict RAW–QDM pairing: identical model membership per scenario/window.
- Baseline 1985–2014; futures 2031–2060, 2041–2070, 2071–2100.
- Calendar harmonized to 365-day no-leap.

## Models
BCC-CSM2-MR, CESM2, CMCC-ESM2, CNRM-CM6-1, CNRM-ESM2-1, GFDL-ESM4, MIROC6, MRI-ESM2-0, NorESM2-MM.
