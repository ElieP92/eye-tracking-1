# analyse_stats.R
# Statistical Analysis for Eye-Tracking Data
# Mixed ANOVA: Group (between) x Valence x Arousal (within)

# ============================================================================
# SETUP
# ============================================================================

# Install packages if needed (to user library)
packages <- c("tidyverse", "afex", "emmeans", "effectsize", "rstatix", "ggpubr", "base64enc")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  user_lib <- Sys.getenv("R_LIBS_USER")
  if (!dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
  install.packages(new_packages, repos = "https://cran.r-project.org", lib = user_lib)
}

library(tidyverse)
library(afex)
library(emmeans)
library(effectsize)
library(rstatix)
library(ggpubr)
library(base64enc)

theme_set(theme_pubr(base_size = 12))

# Paths
script_dir <- tryCatch({
  dirname(rstudioapi::getSourceEditorContext()$path)
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    dirname(normalizePath(sub("--file=", "", file_arg)))
  } else {
    getwd()
  }
})
results_dir <- file.path(dirname(script_dir), "results")
figures_dir <- file.path(results_dir, "figures_R")
report_dir <- file.path(results_dir, "rapport_R")

dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(report_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# LOAD DATA
# ============================================================================

cat("Loading data...\n")

df <- read_delim(file.path(results_dir, "global_preproc.csv"), 
                 delim = ";", locale = locale(decimal_mark = "."),
                 show_col_types = FALSE)

df <- df %>%
  mutate(
    arousal = case_when(
      str_detect(toupper(txt_file), "HACONFIG") ~ "HA",
      str_detect(toupper(txt_file), "LACONFIG|BACONFIG") ~ "LA",
      TRUE ~ NA_character_
    ),
    subject_id = paste(group, subject, sep = "_")
  ) %>%
  filter(!is.na(arousal))

df <- df %>%
  mutate(
    group = factor(group, levels = c("jeunes", "moyen", "age"),
                   labels = c("Young", "Middle-aged", "Older")),
    arousal = factor(arousal, levels = c("HA", "LA"),
                     labels = c("High", "Low"))
  )

cat("Data loaded:", nrow(df), "trials,", n_distinct(df$subject_id), "participants\n")

# ============================================================================
# DATA PREPARATION
# ============================================================================

prepare_fixation_data <- function(data, pair) {
  data %>%
    filter(pair_type == pair, valid_for_fixation) %>%
    pivot_longer(cols = c(n_fix_left, n_fix_right, expansion_left, expansion_right),
                 names_to = c(".value", "side"),
                 names_pattern = "(.+)_(left|right)") %>%
    mutate(valence = if_else(side == "left", val_gauche, val_droit)) %>%
    filter(valence %in% switch(pair,
                               "neg_neu" = c("neg", "neu"),
                               "pos_neu" = c("pos", "neu"),
                               "neg_pos" = c("neg", "pos"))) %>%
    mutate(valence = factor(valence, levels = c("neg", "neu", "pos"),
                            labels = c("Negative", "Neutral", "Positive")))
}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

get_stars <- function(p) {
  case_when(
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ "ns"
  )
}

run_mixed_anova_3way <- function(data, dv) {
  # Aggregate by subject for ANOVA
  agg <- data %>%
    group_by(subject_id, group, arousal, valence) %>%
    summarise(y = mean({{dv}}, na.rm = TRUE), .groups = "drop")
  
  # Check complete cases (all 4 conditions: 2 valence x 2 arousal)
  complete <- agg %>%
    group_by(subject_id) %>%
    filter(n() == 4) %>%
    ungroup()
  
  if (n_distinct(complete$subject_id) < 10) return(NULL)
  
  tryCatch({
    aov_result <- aov_ez(
      id = "subject_id",
      dv = "y",
      data = complete,
      between = "group",
      within = c("valence", "arousal"),
      type = 3
    )
    return(list(aov = aov_result, data = complete, n = n_distinct(complete$subject_id)))
  }, error = function(e) {
    cat("  ANOVA error:", e$message, "\n")
    return(NULL)
  })
}

format_anova_table <- function(aov_result) {
  if (is.null(aov_result)) return(NULL)
  
  tbl <- as.data.frame(aov_result$aov$anova_table)
  tbl$Effect <- rownames(tbl)
  
  tbl %>%
    select(Effect, `num Df`, `den Df`, `F`, `Pr(>F)`, ges) %>%
    rename(df1 = `num Df`, df2 = `den Df`, F_value = `F`, p = `Pr(>F)`, eta2 = ges) %>%
    mutate(sig = get_stars(p))
}

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

plot_3way <- function(data, dv, dv_label, pair_label) {
  summary_data <- data %>%
    group_by(group, arousal, valence) %>%
    summarise(
      mean = mean(y, na.rm = TRUE),
      se = sd(y, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  p <- ggplot(summary_data, aes(x = group, y = mean, fill = valence)) +
    geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7, alpha = 0.85) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se),
                  position = position_dodge(0.8), width = 0.2) +
    facet_wrap(~arousal, ncol = 2) +
    scale_fill_manual(values = c("Negative" = "#c0392b", "Neutral" = "#7f8c8d", "Positive" = "#27ae60")) +
    labs(
      title = paste(dv_label, "-", pair_label, "Pairs"),
      subtitle = "By Age Group, Valence, and Arousal Level",
      x = "Age Group",
      y = dv_label,
      fill = "Valence"
    ) +
    theme(
      legend.position = "bottom",
      strip.background = element_rect(fill = "#34495e"),
      strip.text = element_text(color = "white", face = "bold")
    )
  
  return(p)
}

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

cat("\n========================================\n")
cat("STATISTICAL ANALYSIS WITH R\n")
cat("Mixed ANOVA: Group x Valence x Arousal\n")
cat("========================================\n\n")

all_effects <- tibble()
report_lines <- c(
  "# Eye-Tracking Statistical Analysis - R Report",
  "",
  paste("**Date:**", Sys.Date()),
  paste("**Participants:**", n_distinct(df$subject_id)),
  paste("**Trials:**", nrow(df)),
  "",
  "**Design:** Mixed ANOVA with Group (between) x Valence x Arousal (within)",
  ""
)

figures_list <- list()

# ANALYSIS FOR EACH PAIR TYPE
for (pair in c("neg_neu", "pos_neu", "neg_pos")) {
  pair_label <- toupper(gsub("_", "-", pair))
  cat("Analyzing", pair_label, "pairs...\n")
  
  report_lines <- c(report_lines, "", paste("##", pair_label, "Pairs"), "")
  
  fix_data <- prepare_fixation_data(df, pair)
  
  if (nrow(fix_data) < 100) {
    report_lines <- c(report_lines, "*Insufficient data*", "")
    next
  }
  
  # 1. FIXATIONS ANALYSIS
  cat("  - Fixations...\n")
  report_lines <- c(report_lines, "### Number of Fixations", "")
  
  result <- run_mixed_anova_3way(fix_data, n_fix)
  
  if (!is.null(result)) {
    tbl <- format_anova_table(result)
    
    report_lines <- c(report_lines,
      paste("**Mixed ANOVA (N =", result$n, "participants):**"),
      "",
      "| Effect | df | F | p | eta2 | Sig |",
      "|--------|-----|---|---|------|-----|"
    )
    
    for (i in 1:nrow(tbl)) {
      row <- tbl[i, ]
      report_lines <- c(report_lines,
        sprintf("| %s | %d, %d | %.2f | %.3f | %.3f | %s |",
                row$Effect, as.integer(row$df1), as.integer(row$df2),
                row$F_value, row$p, row$eta2, row$sig))
      
      if (row$p < 0.05) {
        all_effects <- bind_rows(all_effects, tibble(
          Pair = pair_label, Variable = "Fixations", Effect = row$Effect,
          F_value = row$F_value, df = paste0(row$df1, ",", row$df2),
          p = row$p, eta2 = row$eta2, sig = row$sig
        ))
      }
    }
    
    # Plot
    p <- plot_3way(result$data, n_fix, "Number of Fixations", pair_label)
    fig_path <- file.path(figures_dir, paste0("fix_3way_", pair, ".png"))
    ggsave(fig_path, p, width = 10, height = 6, dpi = 150)
    figures_list[[paste0("fix_", pair)]] <- fig_path
    report_lines <- c(report_lines, "", paste0("![Fixations ", pair_label, "](", basename(fig_path), ")"), "")
  }
  
  # 2. EXPANSION ANALYSIS
  cat("  - Expansion...\n")
  report_lines <- c(report_lines, "### Ocular Expansion", "")
  
  result <- run_mixed_anova_3way(fix_data, expansion)
  
  if (!is.null(result)) {
    tbl <- format_anova_table(result)
    
    report_lines <- c(report_lines,
      paste("**Mixed ANOVA (N =", result$n, "participants):**"),
      "",
      "| Effect | df | F | p | eta2 | Sig |",
      "|--------|-----|---|---|------|-----|"
    )
    
    for (i in 1:nrow(tbl)) {
      row <- tbl[i, ]
      report_lines <- c(report_lines,
        sprintf("| %s | %d, %d | %.2f | %.3f | %.3f | %s |",
                row$Effect, as.integer(row$df1), as.integer(row$df2),
                row$F_value, row$p, row$eta2, row$sig))
      
      if (row$p < 0.05) {
        all_effects <- bind_rows(all_effects, tibble(
          Pair = pair_label, Variable = "Expansion", Effect = row$Effect,
          F_value = row$F_value, df = paste0(row$df1, ",", row$df2),
          p = row$p, eta2 = row$eta2, sig = row$sig
        ))
      }
    }
    
    # Plot
    p <- plot_3way(result$data, expansion, "Ocular Expansion (pixels)", pair_label)
    fig_path <- file.path(figures_dir, paste0("exp_3way_", pair, ".png"))
    ggsave(fig_path, p, width = 10, height = 6, dpi = 150)
    figures_list[[paste0("exp_", pair)]] <- fig_path
    report_lines <- c(report_lines, "", paste0("![Expansion ", pair_label, "](", basename(fig_path), ")"), "")
  }
}

# SUMMARY
report_lines <- c(report_lines, "", "## Summary of Significant Effects", "")

if (nrow(all_effects) > 0) {
  report_lines <- c(report_lines,
    "| Pair | Variable | Effect | F | df | p | eta2 | Sig |",
    "|------|----------|--------|---|-----|---|------|-----|"
  )
  
  for (i in 1:nrow(all_effects)) {
    row <- all_effects[i, ]
    report_lines <- c(report_lines,
      sprintf("| %s | %s | %s | %.2f | %s | %.3f | %.3f | %s |",
              row$Pair, row$Variable, row$Effect, row$F_value, row$df, row$p, row$eta2, row$sig))
  }
  
  report_lines <- c(report_lines, "", paste("**Total significant effects:**", nrow(all_effects)))
} else {
  report_lines <- c(report_lines, "*No significant effects detected*")
}

# Save Markdown
writeLines(report_lines, file.path(report_dir, "report_R.md"))
write_csv(all_effects, file.path(report_dir, "significant_effects_R.csv"))

# ============================================================================
# GENERATE HTML REPORT WITH EMBEDDED FIGURES
# ============================================================================

cat("\nGenerating HTML report with embedded figures...\n")

html_head <- '<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Eye-Tracking Analysis - R Statistical Report</title>
    <style>
        @import url("https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap");
        body { font-family: "Source Sans Pro", sans-serif; max-width: 950px; margin: 0 auto; padding: 40px 20px; line-height: 1.6; color: #333; background: #fff; }
        h1 { color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; margin: 30px 0 20px; font-size: 1.8em; }
        h2 { color: #c0392b; border-left: 4px solid #e74c3c; padding-left: 12px; margin: 30px 0 15px; font-size: 1.4em; }
        h3 { color: #7f8c8d; margin: 25px 0 10px; font-size: 1.15em; }
        p { margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }
        th, td { border: 1px solid #ddd; padding: 10px 8px; text-align: left; }
        th { background: #e74c3c; color: white; font-weight: 600; }
        tr:nth-child(even) { background: #f8f9fa; }
        tr:hover { background: #fce4e4; }
        img { max-width: 100%; height: auto; margin: 20px auto; display: block; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .header { text-align: center; padding: 30px; background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { border: none; color: white; margin: 0; font-size: 2em; }
        .header p { margin: 10px 0 0; opacity: 0.9; }
        .sig-3 { color: #c0392b; font-weight: bold; }
        .sig-2 { color: #e67e22; font-weight: bold; }
        .sig-1 { color: #f39c12; font-weight: bold; }
        strong { color: #2c3e50; }
        hr { border: none; border-top: 2px solid #eee; margin: 30px 0; }
        .figure-caption { text-align: center; font-style: italic; color: #666; margin-top: -10px; margin-bottom: 25px; }
        @media print { body { padding: 0; font-size: 11pt; } h1 { page-break-before: always; } h1:first-of-type { page-break-before: avoid; } table, img { page-break-inside: avoid; } }
    </style>
</head>
<body>
<div class="header">
    <h1>Eye-Tracking Statistical Analysis</h1>
    <p>R Report - Mixed ANOVA: Group x Valence x Arousal</p>
</div>
'

html_content <- ""
in_table <- FALSE

for (line in report_lines) {
  # Skip first title (we have header)
  if (grepl("^# Eye-Tracking", line)) next
  
  # Headers
  if (grepl("^## ", line)) {
    if (in_table) { html_content <- paste0(html_content, "</table>\n"); in_table <- FALSE }
    html_content <- paste0(html_content, "<h2>", sub("^## ", "", line), "</h2>\n")
    next
  }
  if (grepl("^### ", line)) {
    if (in_table) { html_content <- paste0(html_content, "</table>\n"); in_table <- FALSE }
    html_content <- paste0(html_content, "<h3>", sub("^### ", "", line), "</h3>\n")
    next
  }
  
  # Tables
  if (grepl("^\\|", line)) {
    if (!in_table) {
      html_content <- paste0(html_content, "<table>\n")
      in_table <- TRUE
    }
    if (grepl("^\\|[-|]+\\|$", line)) next
    
    cells <- strsplit(line, "\\|")[[1]]
    cells <- trimws(cells[cells != ""])
    
    if (grepl("Effect|Pair|Variable", line)) {
      html_content <- paste0(html_content, "<tr>", paste0("<th>", cells, "</th>", collapse = ""), "</tr>\n")
    } else {
      row_html <- "<tr>"
      for (cell in cells) {
        cell <- trimws(cell)
        if (cell == "***") cell <- '<span class="sig-3">***</span>'
        else if (cell == "**") cell <- '<span class="sig-2">**</span>'
        else if (cell == "*") cell <- '<span class="sig-1">*</span>'
        row_html <- paste0(row_html, "<td>", cell, "</td>")
      }
      html_content <- paste0(html_content, row_html, "</tr>\n")
    }
    next
  } else if (in_table) {
    html_content <- paste0(html_content, "</table>\n")
    in_table <- FALSE
  }
  
  # Images - embed as base64
  if (grepl("^!\\[", line)) {
    img_match <- regmatches(line, regexec("!\\[([^]]+)\\]\\(([^)]+)\\)", line))[[1]]
    if (length(img_match) == 3) {
      img_name <- img_match[3]
      img_path <- file.path(figures_dir, img_name)
      if (file.exists(img_path)) {
        img_data <- base64encode(img_path)
        html_content <- paste0(html_content, 
          '<img src="data:image/png;base64,', img_data, '" alt="', img_match[2], '">\n',
          '<p class="figure-caption">', img_match[2], '</p>\n')
      }
    }
    next
  }
  
  # Bold and metadata
  if (grepl("\\*\\*", line)) {
    line <- gsub("\\*\\*([^*]+)\\*\\*", "<strong>\\1</strong>", line)
    html_content <- paste0(html_content, "<p>", line, "</p>\n")
    next
  }
  
  if (line == "" || grepl("^\\*[^*]", line)) next
  
  html_content <- paste0(html_content, "<p>", line, "</p>\n")
}

if (in_table) html_content <- paste0(html_content, "</table>\n")

html_footer <- '
<hr>
<p style="text-align: center; color: #7f8c8d;"><em>Generated with R (afex package) - Mixed ANOVA analysis</em></p>
</body>
</html>'

html_output <- paste0(html_head, html_content, html_footer)
writeLines(html_output, file.path(report_dir, "report_R.html"))

cat("\n========================================\n")
cat("Analysis complete!\n")
cat("Report:", file.path(report_dir, "report_R.md"), "\n")
cat("HTML:", file.path(report_dir, "report_R.html"), "\n")
cat("Figures:", figures_dir, "\n")
cat("Significant effects:", nrow(all_effects), "\n")
cat("========================================\n")
