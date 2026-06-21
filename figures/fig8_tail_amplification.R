# ============================================================
# STEP 7 — Figure 8
# Tail amplification per degree warming
# Main-text manuscript figure (clean summary version)
# A = Delta(hazard) / Delta(T)
# Plot shows QDM − RAW difference in A
# ============================================================

library(readr)
library(dplyr)
library(stringr)
library(ggplot2)

# -----------------------------
# 1) Read data
# -----------------------------
step7 <- read_csv(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP7/step7_A_ensemble_summary.csv",
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

# -----------------------------
# 3) Decode metric labels
# Keep labels simple; explain units in caption
# -----------------------------
step7_main <- step7 %>%
  mutate(
    hazard_id = as.character(hazard_id)
  ) %>%
  filter(
    ssp %in% ssp_levels,
    window %in% window_lvls
  ) %>%
  mutate(
    metric = case_when(
      hazard_type == "exceed_abs" & hazard_id %in% c("35", "35.0") ~ ">35°C exceedance",
      hazard_type == "exceed_abs" & hazard_id %in% c("40", "40.0") ~ ">40°C exceedance",
      hazard_type == "exceed_pct" & hazard_id %in% c("95", "95.0") ~ "P95 exceedance",
      hazard_type == "exceed_pct" & hazard_id %in% c("99", "99.0") ~ "P99 exceedance",
      hazard_type == "heatwave" & hazard_id == "HW95_event_frequency_per_year" ~ "HW95 frequency",
      hazard_type == "heatwave" & hazard_id == "HW95_total_hw_days_per_year" ~ "HW95 days",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(metric)) %>%
  mutate(
    ssp = factor(ssp, levels = ssp_levels, labels = ssp_labels),
    window = factor(window, levels = rev(window_lvls), labels = rev(window_labs)),
    metric = factor(
      metric,
      levels = c(
        ">35°C exceedance",
        ">40°C exceedance",
        "P95 exceedance",
        "P99 exceedance",
        "HW95 frequency",
        "HW95 days"
      )
    ),
    effect = mean_A_diff_QDM_minus_RAW
  )

# -----------------------------
# 4) Figure 8
# -----------------------------
fig_8 <- ggplot(step7_main, aes(y = window, x = effect, color = window)) +
  geom_vline(xintercept = 0, linewidth = 0.55, color = "grey35") +
  geom_segment(
    aes(x = 0, xend = effect, yend = window),
    linewidth = 1.1,
    lineend = "round"
  ) +
  geom_point(size = 3.3) +
  facet_grid(metric ~ ssp, scales = "free_x") +
  scale_color_manual(values = window_cols, name = "Future window") +
  labs(
    x = "Difference in hazard amplification per °C (QDM − RAW)",
    y = NULL
  ) +
  base_theme

# -----------------------------
# 5) Save
# -----------------------------
ggsave(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP7/Figure_8_step7_A_effect_main.png",
  fig_8,
  width = 13.5,
  height = 10.5,
  dpi = 500
)

ggsave(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP7/Figure_8_step7_A_effect_main.pdf",
  fig_8,
  width = 13.5,
  height = 10.5
)

# -----------------------------
# 6) Print
# -----------------------------
print(fig_8)

cat("Saved:\n")
cat(" - Figure_8_step7_A_effect_main.png / .pdf\n")