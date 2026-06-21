# save as: fig2_nature_mean_warming_boxpanel.R
# run: Rscript fig2_nature_mean_warming_boxpanel.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(cowplot)
  library(scales)
})

# -------------------------
# Paths
# -------------------------
BASE   <- "G:/Paper2_updated/LAND_MASKED"
IN_SUM <- file.path(BASE, "ANALYSIS_STEP1", "step1_delta_tas_ensemble_summary.csv")
IN_BYM <- file.path(BASE, "ANALYSIS_STEP1", "step1_delta_tas_by_model.csv")
OUTDIR <- file.path(BASE, "ANALYSIS_STEP1")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

OUT_PNG <- file.path(OUTDIR, "FIG2_mean_warming_preservation_boxpanel.png")
OUT_PDF <- file.path(OUTDIR, "FIG2_mean_warming_preservation_boxpanel.pdf")

# -------------------------
# Load + order
# -------------------------
ens <- read_csv(IN_SUM, show_col_types = FALSE)
bym <- read_csv(IN_BYM, show_col_types = FALSE)

window_order <- c("2031-2060", "2041-2070", "2071-2100")
ssp_order    <- c("ssp126", "ssp245", "ssp585")

ens <- ens %>%
  mutate(
    window = factor(window, levels = window_order, ordered = TRUE),
    ssp    = factor(ssp, levels = ssp_order, ordered = TRUE)
  ) %>%
  arrange(ssp, window)

bym <- bym %>%
  mutate(
    window = factor(window, levels = window_order, ordered = TRUE),
    ssp    = factor(ssp, levels = ssp_order, ordered = TRUE)
  ) %>%
  arrange(ssp, window, model)

ssp_labels <- c(ssp126 = "SSP1–2.6", ssp245 = "SSP2–4.5", ssp585 = "SSP5–8.5")
window_labels <- c("2031-2060"="2031–2060", "2041-2070"="2041–2070", "2071-2100"="2071–2100")

# -------------------------
# Top row data (mean ± half-IQR)
# -------------------------
ens_long <- ens %>%
  transmute(
    ssp, window,
    RAW_mean = mean_delta_RAW,
    QDM_mean = mean_delta_QDM,
    RAW_err  = 0.5 * iqr_delta_RAW,
    QDM_err  = 0.5 * iqr_delta_QDM
  ) %>%
  pivot_longer(
    cols = c(RAW_mean, QDM_mean, RAW_err, QDM_err),
    names_to = c("stream", ".value"),
    names_pattern = "(RAW|QDM)_(mean|err)"
  ) %>%
  mutate(stream = factor(stream, levels = c("RAW", "QDM")))

# -------------------------
# Bottom row box stats (5–95% whiskers)
# -------------------------
box_stats <- bym %>%
  group_by(ssp, window) %>%
  summarise(
    ymin   = quantile(delta_preservation_diff, 0.05, na.rm = TRUE),
    lower  = quantile(delta_preservation_diff, 0.25, na.rm = TRUE),
    middle = quantile(delta_preservation_diff, 0.50, na.rm = TRUE),
    upper  = quantile(delta_preservation_diff, 0.75, na.rm = TRUE),
    ymax   = quantile(delta_preservation_diff, 0.95, na.rm = TRUE),
    .groups = "drop"
  )

# -------------------------
# Theme
# -------------------------
theme_nature <- theme_classic(base_size = 9) +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    
    panel.grid.major.y = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.minor   = element_blank(),
    
    axis.line  = element_line(linewidth = 0.55, color = "grey10"),
    axis.ticks = element_line(linewidth = 0.55, color = "grey10"),
    
    axis.title = element_text(size = 10.5, face = "bold", color = "black"),
    axis.text  = element_text(size = 9.5, face = "bold", color = "black"),
    
    plot.title = element_text(size = 11, face = "bold", hjust = 0.5, color = "black"),
    
    legend.position = "top",
    legend.title = element_blank(),
    legend.text  = element_text(size = 9.5, face = "bold", color = "black"),
    
    plot.margin = margin(6, 12, 6, 12)
  )

COL_RAW <- "#1f77b4"
COL_QDM <- "#ff7f0e"
pd <- position_dodge(width = 0.33)

# -------------------------
# Top panel function
# -------------------------
make_top <- function(ssp_key, show_y = TRUE, show_title = TRUE, show_x = FALSE, show_legend = FALSE) {
  df <- ens_long %>% filter(ssp == ssp_key)
  
  p <- ggplot(df, aes(x = window, y = mean, group = stream)) +
    geom_errorbar(
      aes(ymin = mean - err, ymax = mean + err, color = stream),
      width = 0.13, linewidth = 0.45, position = pd
    ) +
    geom_line(aes(linetype = stream, color = stream), linewidth = 0.95, position = pd) +
    geom_point(aes(shape = stream, color = stream), size = 2.8, stroke = 0.9, position = pd) +
    scale_shape_manual(values = c(RAW = 16, QDM = 15)) +
    scale_linetype_manual(values = c(RAW = "solid", QDM = "22")) +
    scale_color_manual(values = c(RAW = COL_RAW, QDM = COL_QDM)) +
    scale_x_discrete(labels = window_labels) +
    labs(
      title = if (show_title) ssp_labels[[as.character(ssp_key)]] else NULL,
      x = if (show_x) "Future window" else NULL,
      y = if (show_y) expression(bold("Regional land mean warming ("*Delta*"tas, °C)")) else NULL
    ) +
    theme_nature +
    theme(legend.position = if (show_legend) "top" else "none")
  
  if (!show_x) p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  p
}

# -------------------------
# Bottom panel function
# -------------------------
make_bottom <- function(ssp_key, show_y = TRUE, show_x = TRUE) {
  df_pts <- bym %>% filter(ssp == ssp_key)
  df_box <- box_stats %>% filter(ssp == ssp_key)
  
  ggplot() +
    geom_hline(yintercept = 0, linewidth = 0.45, color = "black") +
    geom_boxplot(
      data = df_box,
      aes(x = window, ymin = ymin, lower = lower, middle = middle, upper = upper, ymax = ymax),
      stat = "identity",
      width = 0.60,
      linewidth = 0.40,
      fill = alpha(COL_QDM, 0.22),
      color = COL_QDM
    ) +
    geom_point(
      data = df_pts,
      aes(x = window, y = delta_preservation_diff),
      position = position_jitter(width = 0.10, height = 0, seed = 42),
      size = 1.7, alpha = 0.78, color = COL_RAW
    ) +
    scale_x_discrete(labels = window_labels) +
    labs(
      x = if (show_x) "" else NULL,
      y = if (show_y) expression(bold("Preservation error ("*Delta*"QDM - "*Delta*"RAW, °C)")) else NULL
    ) +
    theme_nature +
    theme(legend.position = "none")
}

# -------------------------
# Shared legend
# -------------------------
legend_plot <- make_top("ssp245", show_y = TRUE, show_title = FALSE, show_x = TRUE, show_legend = TRUE)
legend <- cowplot::get_legend(
  legend_plot + theme(legend.margin = margin(t = -4, r = 0, b = 0, l = 0))
)

# -------------------------
# Panels
# -------------------------
p_top_126 <- make_top("ssp126", TRUE,  TRUE,  FALSE, FALSE)
p_top_245 <- make_top("ssp245", FALSE, TRUE,  FALSE, FALSE)
p_top_585 <- make_top("ssp585", FALSE, TRUE,  FALSE, FALSE)

p_bot_126 <- make_bottom("ssp126", TRUE,  TRUE)
p_bot_245 <- make_bottom("ssp245", FALSE, FALSE)
p_bot_585 <- make_bottom("ssp585", FALSE, FALSE)

top_row <- plot_grid(p_top_126, p_top_245, p_top_585,
                     nrow = 1, align = "hv", axis = "tblr",
                     rel_widths = c(1, 1, 1))

bot_row <- plot_grid(p_bot_126, p_bot_245, p_bot_585,
                     nrow = 1, align = "hv", axis = "tblr",
                     rel_widths = c(1, 1, 1))

# ✅ Combine WITHOUT automatic labels
panels <- plot_grid(
  top_row, bot_row,
  ncol = 1,
  rel_heights = c(1.05, 1.00)
)

# ✅ Add (a) and (b) manually so we can push (b) slightly UP
panels_labeled <- ggdraw(panels) +
  draw_label("(a)", x = 0.01, y = 0.985, hjust = 0, vjust = 1,
             fontface = "bold", size = 13) +
  draw_label("(b)", x = 0.01, y = 0.495, hjust = 0, vjust = 1,  # <-- push UP: increase y slightly (try 0.50 or 0.505)
             fontface = "bold", size = 13)

final <- plot_grid(
  legend,
  panels_labeled,
  ncol = 1,
  rel_heights = c(0.10, 1.0)
)

# -------------------------
# Save
# -------------------------
ggsave(OUT_PNG, plot = final, width = 11.69, height = 8.27, dpi = 450, bg = "white")
ggsave(OUT_PDF, plot = final, width = 11.69, height = 8.27, bg = "white")

cat("Saved:\n")
cat(" -", OUT_PNG, "\n")
cat(" -", OUT_PDF, "\n")