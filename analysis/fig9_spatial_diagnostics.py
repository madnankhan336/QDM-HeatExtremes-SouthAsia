#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

# ===================== PATHS =====================
BASE = r"G:\Paper2_updated\LAND_MASKED\ANALYSIS_STEP8_SPATIAL"
SHAPEFILE = r"F:\Shpfiles\Boundary_of_South_Asia\Boundary of South Asia\Boundary of South Asia.shp"

# ===================== LOAD SHAPEFILE =====================
gdf = gpd.read_file(SHAPEFILE)

if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")

gdf = gdf.to_crs("EPSG:4326")

# ===================== LOAD + CLIP =====================
def load_and_clip(nc_name):
    path = os.path.join(BASE, nc_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    da = xr.open_dataarray(path)

    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})

    da = da.rio.write_crs("EPSG:4326", inplace=True)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da = da.rio.clip(gdf.geometry, gdf.crs, drop=True)

    for dim in ["model", "quantile", "bnds", "time", "height", "spatial_ref"]:
        if dim in da.dims:
            da = da.mean(dim=dim, skipna=True)

    for coord in ["model", "quantile", "bnds", "time", "height", "spatial_ref"]:
        if coord in da.coords:
            da = da.drop_vars(coord)

    return da.squeeze()

# ===================== DATA =====================
scenarios = ["ssp126", "ssp245", "ssp585"]
periods = ["mid", "late"]

data = {}

for ssp in scenarios:
    for p in periods:
        data[f"diff_days40_{p}_{ssp}"] = load_and_clip(f"diff_days40_{p}_{ssp}.nc")
        data[f"diff_hw_mean_duration_{p}_{ssp}"] = load_and_clip(f"diff_hw_mean_duration_{p}_{ssp}.nc")
        data[f"ratio_iqr_q99_{p}_{ssp}"] = load_and_clip(f"ratio_iqr_q99_{p}_{ssp}.nc")

# ===================== FIGURE — FULL BIG LANDSCAPE PAGE =====================
fig, axes = plt.subplots(
    3, 6,
    figsize=(31.0, 14.5),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

row_titles = [
    "Δ Days >40°C\n(QDM − RAW)",
    "Δ Mean HW40 Duration\n(QDM − RAW)",
    "IQR Ratio (QDM/RAW)\nat 99th percentile"
]

scenario_labels = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5"
}

period_labels = {
    "mid": "2041–2070",
    "late": "2071–2100"
}

col_titles = []

for ssp in scenarios:
    for p in periods:
        col_titles.append(f"{scenario_labels[ssp]}\n{period_labels[p]}")

# ===================== COLORS & NORMALIZATION =====================
cmaps = ["RdYlBu_r", "RdYlBu_r", "RdYlBu_r"]

norms = [
    TwoSlopeNorm(vmin=-25, vcenter=0.001, vmax=25),
    TwoSlopeNorm(vmin=-4, vcenter=0.001, vmax=4),
    TwoSlopeNorm(vmin=0.5, vcenter=1.001, vmax=1.8)
]

row_mappables = [None, None, None]

# Fewer grid ticks
xticks = [60, 70, 80, 90]
yticks = [10, 20, 30]

# ===================== PLOTTING =====================
for r in range(3):
    for c in range(6):

        ax = axes[r, c]

        ssp = scenarios[c // 2]
        p = periods[c % 2]

        if r == 0:
            da = data[f"diff_days40_{p}_{ssp}"]
        elif r == 1:
            da = data[f"diff_hw_mean_duration_{p}_{ssp}"]
        else:
            da = data[f"ratio_iqr_q99_{p}_{ssp}"]

        im = ax.pcolormesh(
            da["lon"],
            da["lat"],
            da.values,
            cmap=cmaps[r],
            norm=norms[r],
            transform=ccrs.PlateCarree(),
            shading="auto"
        )

        row_mappables[r] = im

        gdf.boundary.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            color="black",
            linewidth=0.85
        )

        ax.set_extent([58, 98, 2, 38], crs=ccrs.PlateCarree())

        # Less dense grid
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())

        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

        ax.grid(
            True,
            linewidth=0.16,
            color="gray",
            alpha=0.20,
            linestyle="--"
        )

        # Only left latitude labels
        if c != 0:
            ax.set_yticklabels([])
        else:
            for label in ax.get_yticklabels():
                label.set_rotation(90)
                label.set_fontsize(10.0)
                label.set_fontweight("bold")

        # Only bottom longitude labels
        if r != 2:
            ax.set_xticklabels([])
        else:
            for label in ax.get_xticklabels():
                label.set_fontsize(10.0)
                label.set_fontweight("bold")

        # Column titles
        if r == 0:
            ax.set_title(
                col_titles[c],
                fontsize=14.0,
                fontweight="bold",
                pad=6
            )

# ===================== LAYOUT — BIG MAPS, FULL WIDTH, TIGHT ROWS =====================
plt.subplots_adjust(
    left=0.055,
    right=0.895,
    bottom=0.040,
    top=0.965,
    wspace=0.006,
    hspace=-0.22
)

fig.canvas.draw()

# ===================== ROW LABELS =====================
for r in range(3):

    row_y0 = min(axes[r, c].get_position().y0 for c in range(6))
    row_y1 = max(axes[r, c].get_position().y1 for c in range(6))
    row_yc = (row_y0 + row_y1) / 2

    fig.text(
        0.030,
        row_yc,
        row_titles[r],
        rotation=90,
        va="center",
        ha="center",
        fontsize=13.8,
        fontweight="bold"
    )

# ===================== COLORBARS — CLOSE TO FRAME BUT NOT TOUCHING EACH OTHER =====================
cbar_x = 0.905
cbar_w = 0.012

for r in range(3):

    row_y0 = min(axes[r, c].get_position().y0 for c in range(6))
    row_y1 = max(axes[r, c].get_position().y1 for c in range(6))
    row_h = row_y1 - row_y0

    # Shorten each colorbar slightly so legends do not touch between rows
    gap = 0.030 * row_h
    cax_y0 = row_y0 + gap
    cax_h = row_h - 2 * gap

    cax = fig.add_axes([cbar_x, cax_y0, cbar_w, cax_h])
    cbar = fig.colorbar(row_mappables[r], cax=cax)

    if r == 0:
        cbar.set_label("Δ Days/year", fontsize=12.0, fontweight="bold")
    elif r == 1:
        cbar.set_label("Δ Days", fontsize=12.0, fontweight="bold")
    else:
        cbar.set_label("Ratio", fontsize=12.0, fontweight="bold")

    cbar.ax.tick_params(labelsize=10.0, width=1.2)

    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

# ===================== SAVE =====================
out_png = os.path.join(BASE, "Figure9_Spatial_AllScenarios_FULL_PAGE_LARGE_MAPS.png")
out_pdf = os.path.join(BASE, "Figure9_Spatial_AllScenarios_FULL_PAGE_LARGE_MAPS.pdf")

fig.savefig(
    out_png,
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02
)

fig.savefig(
    out_pdf,
    bbox_inches="tight",
    pad_inches=0.02
)

print("✅ FULL-PAGE LARGE MAP FIGURE SAVED:")
print(out_png)
print(out_pdf)

plt.show()