# ============================================================
# Q1-style manuscript figures for Step 4 and Step 5
# Same size for Figure 5A and Figure 5B
# Units in superscript
# Supplementary y-axis improved
# Zero line removed so it matches other grid lines
# ============================================================

library(readr)
library(dplyr)
library(stringr)
library(ggplot2)

# -----------------------------
# 1) Read data
# -----------------------------
step4 <- read_csv(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP4/step4_heatwave_ensemble_summary.csv",
  show_col_types = FALSE
)

step5 <- read_csv(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP5/step5_temporal_ensemble_summary.csv",
  show_col_types = FALSE
)

# -----------------------------
# 2) Common labels and palette
# -----------------------------
ssp_levels  <- c("ssp126", "ssp245", "ssp585")
ssp_labels  <- c("SSP1-2.6", "SSP2-4.5", "SSP5-8.5")

window_lvls <- c("2031-2060", "2041-2070", "2071-2100")
window_labs <- c("2031–2060", "2041–2070", "2071–2100")

window_cols <- c(
  "2031–2060" = "#4C78A8",
  "2041–2070" = "#54A24B",
  "2071–2100" = "#E45756"
)

base_theme <- theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "grey85", linewidth = 0.4),
    strip.text = element_text(face = "bold", size = 11),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(color = "black", size = 11),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10.5),
    legend.box = "horizontal",
    plot.margin = margin(t = 8, r = 12, b = 8, l = 12)
  )

supp_theme <- theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "grey85", linewidth = 0.4),
    panel.grid.major.y = element_line(color = "grey85", linewidth = 0.4),
    strip.text = element_text(face = "bold", size = 11),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(color = "black", size = 11),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10.5),
    legend.box = "horizontal",
    plot.margin = margin(t = 8, r = 12, b = 8, l = 12)
  )

# ============================================================
# FIGURE 5A
# HW95 persistence
# ============================================================

step4_main <- step4 %>%
  filter(
    heatwave == "HW95",
    metric %in% c(
      "event_frequency_per_year",
      "mean_duration_days",
      "total_hw_days_per_year"
    ),
    ssp %in% ssp_levels,
    window %in% window_lvls
  ) %>%
  mutate(
    ssp = factor(ssp, levels = ssp_levels, labels = ssp_labels),
    window = factor(window, levels = rev(window_lvls), labels = rev(window_labs)),
    metric = recode(
      metric,
      event_frequency_per_year = "HW95~frequency~(yr^{-1})",
      mean_duration_days = "HW95~duration~(days)",
      total_hw_days_per_year = "HW95~days~(days~yr^{-1})"
    ),
    effect = mean_delta_diff_QDM_minus_RAW
  )

fig_5A <- ggplot(step4_main, aes(y = window, x = effect, color = window)) +
  geom_vline(xintercept = 0, linewidth = 0.55, color = "grey35") +
  geom_segment(
    aes(x = 0, xend = effect, yend = window),
    linewidth = 1.1,
    lineend = "round"
  ) +
  geom_point(size = 3.3) +
  facet_grid(metric ~ ssp, scales = "free_x", labeller = labeller(metric = label_parsed)) +
  scale_color_manual(values = window_cols, name = "Future window") +
  labs(
    x = "QDM − RAW difference",
    y = NULL
  ) +
  base_theme

ggsave(
  "Figure_5A_HW95_persistence_clean.png",
  fig_5A,
  width = 13.5,
  height = 8.0,
  dpi = 500
)

ggsave(
  "Figure_5A_HW95_persistence_clean.pdf",
  fig_5A,
  width = 13.5,
  height = 8.0
)

# ============================================================
# FIGURE 5B
# P95 persistence/clustering
# ============================================================

step5_main <- step5 %>%
  filter(
    ssp %in% ssp_levels,
    window %in% window_lvls,
    metric %in% c("run95_exceed_days_per_year", "run95_mean_run_length")
  ) %>%
  mutate(
    ssp = factor(ssp, levels = ssp_levels, labels = ssp_labels),
    window = factor(window, levels = rev(window_lvls), labels = rev(window_labs)),
    metric = recode(
      metric,
      run95_exceed_days_per_year = "P95~exceedance~days~(days~yr^{-1})",
      run95_mean_run_length = "P95~persistence~(days)"
    ),
    effect = mean_diff_qdm_minus_raw
  )

fig_5B <- ggplot(step5_main, aes(y = window, x = effect, color = window)) +
  geom_vline(xintercept = 0, linewidth = 0.55, color = "grey35") +
  geom_segment(
    aes(x = 0, xend = effect, yend = window),
    linewidth = 1.1,
    lineend = "round"
  ) +
  geom_point(size = 3.3) +
  facet_grid(metric ~ ssp, scales = "free_x", labeller = labeller(metric = label_parsed)) +
  scale_color_manual(values = window_cols, name = "Future window") +
  labs(
    x = "QDM − RAW difference",
    y = NULL
  ) +
  base_theme

ggsave(
  "Figure_5B_P95_persistence_clean.png",
  fig_5B,
  width = 13.5,
  height = 8.0,
  dpi = 500
)

ggsave(
  "Figure_5B_P95_persistence_clean.pdf",
  fig_5B,
  width = 13.5,
  height = 8.0
)

# ============================================================
# FIGURE S1
# ACF preservation
# ============================================================

step5_acf <- step5 %>%
  filter(
    ssp %in% ssp_levels,
    window %in% window_lvls,
    str_detect(metric, "^acf_lag[1-7]$")
  ) %>%
  mutate(
    ssp = factor(ssp, levels = ssp_levels, labels = ssp_labels),
    window = factor(window, levels = window_lvls, labels = window_labs),
    lag = as.integer(str_extract(metric, "\\d+")),
    effect = mean_diff_qdm_minus_raw
  )

fig_s1 <- ggplot(step5_acf, aes(x = lag, y = effect, color = window, group = window)) +
  geom_line(linewidth = 1.1, lineend = "round") +
  geom_point(size = 2.6) +
  facet_wrap(~ ssp, ncol = 1) +
  scale_color_manual(values = window_cols, name = "Future window") +
  scale_x_continuous(breaks = 1:7) +
  scale_y_continuous(
    limits = c(-0.026, 0.006),
    breaks = c(-0.025, -0.020, -0.015, -0.010, -0.005, 0.000, 0.005)
  ) +
  labs(
    x = "Lag (days)",
    y = "QDM − RAW difference"
  ) +
  supp_theme

ggsave(
  "Figure_S1_ACF_preservation_clean.png",
  fig_s1,
  width = 8.6,
  height = 10.8,
  dpi = 500
)

ggsave(
  "Figure_S1_ACF_preservation_clean.pdf",
  fig_s1,
  width = 8.6,
  height = 10.8
)

# -----------------------------
# 3) Print to viewer
# -----------------------------
print(fig_5A)
print(fig_5B)
print(fig_s1)

cat("Saved:\n")
cat(" - Figure_5A_HW95_persistence_clean.png / .pdf\n")
cat(" - Figure_5B_P95_persistence_clean.png / .pdf\n")
cat(" - Figure_S1_ACF_preservation_clean.png / .pdf\n")

