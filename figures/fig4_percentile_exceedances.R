# =========================================================
# Figure 4: Percentile exceedance changes (RAW vs QDM)
# Polished 6-panel grouped bar plot (journal-style)
# Rows = P99 (top), P95 (bottom)
# Cols = SSP1-2.6, SSP2-4.5, SSP5-8.5
# FINAL version matching Figure 3
# =========================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(forcats)
library(grid)
library(ggh4x)

# ---------------------------
# 1. Paths
# ---------------------------
base_dir <- "G:/Paper2_updated/LAND_MASKED/ANALYSIS_STEP3"
infile   <- file.path(base_dir, "step3_percentile_exceedance_ensemble_summary.csv")
out_png  <- file.path(base_dir, "Figure4_percentile_exceedance_changes_polished_final.png")
out_pdf  <- file.path(base_dir, "Figure4_percentile_exceedance_changes_polished_final.pdf")

# ---------------------------
# 2. Read data
# ---------------------------
df <- read_csv(infile, show_col_types = FALSE)

# ---------------------------
# 3. Prepare data
# ---------------------------
plot_df <- df %>%
  mutate(
    percentile = factor(
      percentile,
      levels = c(99, 95),
      labels = c("P99", "P95")
    ),
    ssp = factor(
      ssp,
      levels = c("ssp126", "ssp245", "ssp585"),
      labels = c("SSP1-2.6", "SSP2-4.5", "SSP5-8.5")
    ),
    window = factor(
      window,
      levels = c("2031-2060", "2041-2070", "2071-2100"),
      labels = c("2031–2060", "2041–2070", "2071–2100")
    )
  ) %>%
  pivot_longer(
    cols = c(
      mean_delta_days_RAW,
      mean_delta_days_QDM,
      iqr_delta_days_RAW,
      iqr_delta_days_QDM
    ),
    names_to = c(".value", "stream"),
    names_pattern = "(mean_delta_days|iqr_delta_days)_(RAW|QDM)"
  ) %>%
  mutate(
    stream = factor(stream, levels = c("RAW", "QDM")),
    ymin = pmax(0, mean_delta_days - iqr_delta_days / 2),
    ymax = mean_delta_days + iqr_delta_days / 2
  )

# ---------------------------
# 4. Position dodge
# ---------------------------
pd <- position_dodge(width = 0.50)

# ---------------------------
# 5. Plot
# ---------------------------
p <- ggplot(
  plot_df,
  aes(
    x = window,
    y = mean_delta_days,
    fill = stream
  )
) +
  geom_col(
    position = pd,
    width = 0.34,
    color = "black",
    linewidth = 0.25
  ) +
  geom_errorbar(
    aes(ymin = ymin, ymax = ymax),
    position = pd,
    width = 0.10,
    linewidth = 0.60
  ) +
  ggh4x::facet_grid2(
    rows = vars(percentile),
    cols = vars(ssp),
    scales = "free_y",
    independent = "y",
    axes = "all"
  ) +
  scale_fill_manual(
    values = c(
      "RAW" = "#9e9e9e",
      "QDM" = "#4daf8a"
    )
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.10))
  ) +
  labs(
    x = "",
    y = expression(bold("Change in exceedance days (" * days ~ yr^{-1} * ")")),
    fill = NULL
  ) +
  coord_cartesian(clip = "off") +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "top",
    legend.direction = "horizontal",
    legend.text = element_text(face = "bold", size = 12, color = "black"),
    legend.title = element_text(face = "bold", size = 12, color = "black"),
    
    strip.background = element_rect(
      fill = "grey96",
      color = "black",
      linewidth = 0.35
    ),
    strip.text.x = element_text(
      face = "bold",
      size = 12,
      color = "black"
    ),
    strip.text.y.right = element_text(
      face = "bold",
      size = 12,
      color = "black",
      angle = 270
    ),
    
    axis.title = element_text(
      face = "bold",
      size = 12,
      color = "black"
    ),
    
    axis.text.x = element_text(
      face = "bold",
      size = 10.5,
      color = "black"
    ),
    axis.text.y = element_text(
      face = "bold",
      size = 10.5,
      color = "black"
    ),
    
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(
      color = "grey90",
      linewidth = 0.30
    ),
    
    panel.spacing.x = unit(0.60, "lines"),
    panel.spacing.y = unit(0.60, "lines"),
    
    plot.margin = margin(8, 18, 8, 8)
  )

# ---------------------------
# 6. Save
# ---------------------------
ggsave(out_png, plot = p, width = 12, height = 7, dpi = 600, bg = "white")
ggsave(out_pdf, plot = p, width = 12, height = 7, bg = "white")

print(p)