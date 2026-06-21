# ============================================================
# STEP 6 — Figure 7
# Same manuscript style as Figure 5 / Figure S1
# Main figure: R_IQR across quantiles
# Supplement: Table S6
# SSP labels shown on the RIGHT side, vertically
# Added ruler-like y-axis detail with minor breaks/grid/ticks
# ============================================================

library(readr)
library(dplyr)
library(ggplot2)

# -----------------------------
# 1) Read data
# -----------------------------
step6_unc <- read_csv(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP6/step6_uncertainty_by_quantile.csv",
  show_col_types = FALSE
)

step6_tail <- read_csv(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP6/step6_tail_reshaping_index.csv",
  show_col_types = FALSE
)

# -----------------------------
# 2) Common labels and palette
# -----------------------------
ssp_levels  <- c("ssp126", "ssp245", "ssp585")
ssp_labels  <- c("SSP1-2.6", "SSP2-4.5", "SSP5-8.5")

window_lvls <- c("2031-2060", "2041-2070", "2071-2100")
window_labs <- c("2031–2060", "2041–2070", "2071–2100")

q_levels <- c("q50", "q90", "q95", "q99")
q_labels <- c("0.50", "0.90", "0.95", "0.99")

window_cols <- c(
  "2031–2060" = "#4C78A8",
  "2041–2070" = "#54A24B",
  "2071–2100" = "#E45756"
)

base_theme <- theme_minimal(base_size = 13) +
  theme(
    panel.grid.major.y = element_line(color = "grey85", linewidth = 0.40),
    panel.grid.minor.y = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    
    strip.text = element_text(face = "bold", size = 11),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(color = "black", size = 11),
    
    axis.ticks.y = element_line(color = "black", linewidth = 0.35),
    axis.ticks.x = element_line(color = "black", linewidth = 0.35),
    axis.ticks.length = unit(2.5, "pt"),
    
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10.5),
    legend.box = "horizontal",
    
    plot.margin = margin(t = 8, r = 12, b = 8, l = 12)
  )

# -----------------------------
# 3) Prepare Figure 7 data
# -----------------------------
step6_main <- step6_unc %>%
  filter(
    ssp %in% ssp_levels,
    window %in% window_lvls,
    quantile %in% q_levels
  ) %>%
  mutate(
    ssp = factor(ssp, levels = ssp_levels, labels = ssp_labels),
    window = factor(window, levels = window_lvls, labels = window_labs),
    quantile = factor(quantile, levels = q_levels, labels = q_labels),
    x_base = c("0.50" = 1, "0.90" = 2, "0.95" = 3, "0.99" = 4)[as.character(quantile)],
    x_off  = c("2031–2060" = -0.16, "2041–2070" = 0, "2071–2100" = 0.16)[as.character(window)],
    x = x_base + x_off,
    y = R_IQR_QDM_over_RAW
  )

# -----------------------------
# 4) Figure 7
# -----------------------------
fig_7 <- ggplot(step6_main, aes(x = x, y = y, color = window)) +
  geom_hline(yintercept = 1, linewidth = 0.55, color = "grey35") +
  geom_segment(
    aes(xend = x, y = 0, yend = y),
    linewidth = 1.1,
    lineend = "round"
  ) +
  geom_point(size = 3.0, stroke = 0.45) +
  facet_grid(ssp ~ .) +
  scale_x_continuous(
    breaks = 1:4,
    labels = q_labels,
    expand = expansion(mult = c(0.08, 0.08))
  ) +
  scale_y_continuous(
    limits = c(0, 1.02),
    breaks = c(0.00, 0.25, 0.50, 0.75, 1.00),
    minor_breaks = seq(0, 1.00, by = 0.05),
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_color_manual(values = window_cols, name = "Future window") +
  labs(
    x = "Quantile",
    y = "Spread transformation ratio, R[IQR]"
  ) +
  base_theme +
  theme(
    panel.border = element_rect(color = "grey82", fill = NA, linewidth = 0.7),
    panel.spacing.y = unit(10, "pt"),
    strip.background = element_blank(),
    strip.placement = "outside",
    strip.text.y.left = element_blank(),
    strip.text.y.right = element_text(face = "bold", size = 11, angle = 270)
  )

ggsave(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP6/Figure_7_step6_RIQR_main.png",
  fig_7,
  width = 8.0,
  height = 8.4,
  dpi = 500
)

ggsave(
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP6/Figure_7_step6_RIQR_main.pdf",
  fig_7,
  width = 8.0,
  height = 8.4
)

# -----------------------------
# 5) Table S6
# -----------------------------
table_s6 <- step6_tail %>%
  filter(
    ssp %in% ssp_levels,
    window %in% window_lvls
  ) %>%
  mutate(
    Scenario = factor(ssp, levels = ssp_levels, labels = ssp_labels),
    Window   = factor(window, levels = window_lvls, labels = window_labs)
  ) %>%
  arrange(Scenario, Window) %>%
  transmute(
    Scenario,
    Window,
    `R_IQR(q50)` = round(R_IQR_q50, 3),
    `R_IQR(q99)` = round(R_IQR_q99, 3),
    `Tail reshaping index (S)` = round(Tail_reshaping_index_S, 3)
  )

write_csv(
  table_s6,
  "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP6/Table_S6_step6_tail_index.csv"
)

# -----------------------------
# 6) Print
# -----------------------------
print(fig_7)

cat("Saved:\n")
cat(" - Figure_7_step6_RIQR_main.png / .pdf\n")
cat(" - Table_S6_step6_tail_index.csv\n")
