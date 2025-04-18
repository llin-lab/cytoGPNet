# Load libraries
library(flowCore)
library(diffcyt)
library(tidyverse)

# Source plotting scripts
source("~/Desktop/AutoEncoderCyTOF/TOP1501/plotHeatmap_DA.R")
source("~/Desktop/AutoEncoderCyTOF/TOP1501/plotHeatmap_DS.R")

#-------------------------------#
# Convert DataFrame or List to flowFrame or list of flowFrames
#-------------------------------#
DFtoFF <- function(DF) {
  if (is.data.frame(DF)) {
    return(flowFrame(as.matrix(DF)))
  } 
  if (is.list(DF)) {
    frameList <- vector("list", length(DF))
    for (i in seq_along(DF)) {
      if (is.data.frame(DF[[i]])) {
        frameList[[i]] <- flowFrame(as.matrix(DF[[i]]))
        names(frameList)[i] <- names(DF)[i]
      } else {
        warning(paste("Object at index", i, "not of type data.frame"))
      }
    }
    return(frameList)
  }
  stop("Object is not of type data.frame or list")
}

#-------------------------------#
# Load and preprocess Baseline data
#-------------------------------#
baseline <- read_csv("~/Desktop/TOP1501/Data_revised4/calibrated_BiexTransform_concat_by_batch_Baseline.csv")

colnames(baseline)[1:25] <- c("CD39", "488_610/20_bD-A", "488_660/40_bC-A", "CD366",
                                      "488_820/60_bA-A", "Granzyme B",
                                      "405_610/20_vE-A", "CD127", "CD103", "CD4", "405_800/60_vA-A",
                                      "CD8", "CD45RA", "CD25", "HLADR", "Ki67", "CD38",
                                      "KLRG1", "CD28", "CD197", "532_575/25_gE-A", "Tbet",
                                      "CD152", "EOMES", "CD279")
baseline_labels <- read_csv("~/Desktop/TOP1501/Data_revised4/TOP1501_Baseline_label.csv")
baseline_labels <- baseline_labels[,7:(ncol(baseline_labels))]
colnames(baseline_labels)[17:19] <- c("CD4+ Q1", "CD4+ Q3", "CD4+ Q4")
colnames(baseline_labels)[45:47] <- c("CD8+ Q1", "CD8+ Q3", "CD8+ Q4")
baseline_labels %>% mutate(Sample = paste(str_split(Sample, pattern = "-")[[1]][1], str_split(Sample, pattern = "-")[[1]][2], sep = "_")) -> baseline_labels


#-------------------------------#
# Load and preprocess Treatment data
#-------------------------------#
treatment <- read_csv("~/Desktop/TOP1501/Data_revised4/calibrated_BiexTransform_concat_by_batch_Treatment.csv")

colnames(treatment)[1:25] <- c("CD39", "488_610/20_bD-A", "488_660/40_bC-A", "CD366",
                                       "488_820/60_bA-A", "Granzyme B",
                                       "405_610/20_vE-A", "CD127", "CD103", "CD4", "405_800/60_vA-A",
                                       "CD8", "CD45RA", "CD25", "HLADR", "Ki67", "CD38",
                                       "KLRG1", "CD28", "CD197", "532_575/25_gE-A", "Tbet",
                                       "CD152", "EOMES", "CD279")
treatment_labels <- read_csv("~/Desktop/TOP1501/Data_revised4/TOP1501_Treatment_label.csv")
treatment_labels <- treatment_labels[,7:(ncol(treatment_labels))]
colnames(treatment_labels)[17:19] <- c("CD4+ Q1", "CD4+ Q3", "CD4+ Q4")
colnames(treatment_labels)[45:47] <- c("CD8+ Q1", "CD8+ Q3", "CD8+ Q4")
treatment_labels %>% mutate(Sample = paste(str_split(Sample, pattern = "-")[[1]][1], str_split(Sample, pattern = "-")[[1]][2], sep = "_")) -> treatment_labels


#-------------------------------#
# Remove unwanted channels
#-------------------------------#
exclude_channels <- c("488_610/20_bD-A", "488_660/40_bC-A", "488_820/60_bA-A",
                      "405_610/20_vE-A", "405_800/60_vA-A", "532_575/25_gE-A")

baseline_new <- baseline %>% select(-all_of(exclude_channels))
treatment_new <- treatment %>% select(-all_of(exclude_channels))

#-------------------------------#
# Write FCS files
#-------------------------------#
write_fcs_files <- function(data, suffix) {
  samples <- unique(data$Sample)
  for (sample in samples) {
    temp <- data %>%
      filter(Sample == sample) %>%
      select(-Sample, -Responders)
    ff <- DFtoFF(temp)
    out_path <- paste0("~/Desktop/AutoEncoderCyTOF/TOP1501/FCS/", sample, "_", suffix, ".fcs")
    write.FCS(ff, out_path)
  }
}

write_fcs_files(baseline_new, "Baseline")
write_fcs_files(treatment_new, "Treatment")

#-------------------------------#
# Read FCS files and create experiment info
#-------------------------------#
files <- list.files("~/Desktop/AutoEncoderCyTOF/TOP1501/FCS/", pattern = "_Baseline.fcs$", full.names = TRUE)
d_flowSet <- read.flowSet(files, transformation = FALSE, truncate_max_range = FALSE)

filenames <- as.character(pData(d_flowSet)$name)
sample_id <- str_remove(filenames, ".fcs")
group_id <- factor(str_extract(sample_id, "Baseline|Treatment"), levels = c("Baseline", "Treatment"))
patient_id <- str_remove_all(filenames, "_(Baseline|Treatment)\\.fcs")

experiment_info <- data.frame(group_id, patient_id, sample_id, stringsAsFactors = FALSE)

#-------------------------------#
# Marker info
#-------------------------------#
expr_df <- baseline_new %>% select(-Sample, -Responders, -all_of(exclude_channels))
temp <- expr_df[1, , drop = FALSE]  # dummy for colnames

cols_markers <- seq_along(temp)
cols_lineage <- which(colnames(temp) %in% c("CD4", "CD8", "CD45RA", "CD103", "CD25", "CD197", "HLADR", "CD38"))
cols_func <- which(colnames(temp) %in% c("CD39", "CD366", "Granzyme B", "CD127", "Ki67", "KLRG1", "CD28", "Tbet", "CD152", "EOMES", "CD279"))

channel_name <- colnames(d_flowSet)
marker_name <- gsub("\\(.*$", "", channel_name)

marker_class <- rep("none", length(channel_name))
marker_class[cols_lineage] <- "type"
marker_class[cols_func] <- "state"
marker_class <- factor(marker_class, levels = c("type", "state", "none"))

marker_info <- data.frame(channel_name, marker_name, marker_class, stringsAsFactors = FALSE)

#-------------------------------#
# diffcyt analysis
#-------------------------------#
design <- createDesignMatrix(experiment_info, cols_design = c("group_id", "patient_id"))

metadata <- baseline_new %>%
  select(Sample, Responders) %>%
  distinct() %>%
  mutate(outcome = ifelse(Responders == "Major", 1, 0))

contrast <- createContrast(c(0, 1, metadata$outcome[-1]))

# Prepare data for diffcyt
d_se <- prepareData(d_flowSet, experiment_info, marker_info)
d_se <- transformData(d_se)
d_se <- generateClusters(d_se, seed_clustering = 1)

# Calculate counts and medians
d_counts <- calcCounts(d_se)
d_medians <- calcMedians(d_se)

# Differential abundance testing
res_DA <- testDA_edgeR(d_counts, design, contrast)

# Display table of results for top DA clusters
topTable(res_DA, format_vals = TRUE)

# Calculate the number of significant DA clusters at 5% false discovery rate (FDR)
threshold <- 0.05
significant_DA <- table(topTable(res_DA, all = TRUE)$p_adj <= threshold)

# Print number of significant DA clusters
print(significant_DA)

#-------------------------------#
# Differential States (DS) Analysis within Clusters
#-------------------------------#

# Test for Differential States (DS) within clusters
res_DS <- testDS_limma(d_counts, d_medians, design, contrast, plot = TRUE)

# Display table of results for top DS cluster-marker combinations
topTable(res_DS, format_vals = TRUE)

#-------------------------------#
# Heatmap for Top Detected DA Clusters
#-------------------------------#

# Prepare output list for DA analysis results
out_DA <- list(res = res_DA, d_se = d_se, d_counts = d_counts)

# Define custom sample order (alternating)
sample_order <- c(seq(1, 58, by = 2), seq(2, 58, by = 2))

# Plot heatmap for DA clusters
plotHeatmap(out = out_DA, analysis_type = "DA", sample_order = sample_order)

# Alternatively, use plotHeatmap_DA function for a different style of DA heatmap
plotHeatmap_DA(res = res_DA, d_se = d_se, d_counts = d_counts, 
               analysis_type = "DA", sample_order = sample_order)

# Save heatmap plot as PNG file
ggsave("~/Desktop/AutoEncoderCyTOF/TOP1501/FCS/Diffcyt2.png", 
       device = "png", width = 15.5, height = 5, units = "in", dpi = 1000)

plotHeatmap_DS(res = res_DS, analysis_type = "DS", d_se = d_se, d_counts = d_counts, sample_order = sample_order)
ggsave("~/Desktop/AutoEncoderCyTOF/TOP1501/FCS/Diffcyt1.png", device = "png", width = 15.5, height = 7.5, units = "in", dpi=1000)


# Load required libraries
library(glmnet)

# Load expression and metadata
TOP1501 <- read_csv("~/Desktop/AutoEncoderCyTOF/TOP1501/TOP1501_Baseline_Treatment_arcsinh.csv") %>% 
  mutate(patient = Sample)

TOP1501_metadata <- read_csv("~/Desktop/AutoEncoderCyTOF/TOP1501/TOP1501_metadata.csv") %>% 
  mutate(patient = Sample)

# Run proportional test
set.seed(123)
result <- proportional_test(
  dataframe = as.data.frame(TOP1501),
  metadata = TOP1501_metadata,
  columns = c(1,4,6,8,9,10,12:20, 22:25),
  save_dir = "~/Desktop/AutoEncoderCyTOF/TOP1501/"
)

# Initialize matrix to store gating proportions
num_clusters <- ncol(baseline_labels) - 1
gating_prop <- matrix(0, ncol = num_clusters * 2, nrow = nrow(TOP1501_metadata))

# Compute counts and proportions for each cluster
for (i in 1:num_clusters) {
  cluster_name <- colnames(baseline_labels)[i]
  
  # Baseline
  temp_base <- baseline_labels[, c(cluster_name, "Sample")] %>%
    group_by(Sample) %>%
    summarise(n = sum(.data[[cluster_name]]), p = n(), .groups = "drop") %>%
    mutate(prop = n / p)
  
  gating_prop[, i] <- temp_base$n
  gating_prop[, i + num_clusters] <- temp_base$prop
  
  # Treatment
  temp_treat <- treatment_labels[, c(cluster_name, "Sample")] %>%
    group_by(Sample) %>%
    summarise(n = sum(.data[[cluster_name]]), p = n(), .groups = "drop") %>%
    mutate(prop = n / p)
  
  gating_prop[, i + num_clusters] <- temp_treat$prop
}

# Final gating data frame with metadata
gating_prop <- as.data.frame(gating_prop)
gating_prop$group <- TOP1501_metadata$group
gating_prop$outcome <- TOP1501_metadata$outcome

#-------------------------------------------#
# LASSO Logistic Regression
#-------------------------------------------#

result1 <- c()
truth <- c()
group <- c()

set.seed(123)
for (i in 1:5) {
  train_data <- gating_prop %>% filter(group != i) %>% select(-group)
  test_data  <- gating_prop %>% filter(group == i) %>% select(-group)
  
  truth <- c(truth, test_data$outcome)
  group <- c(group, rep(i, nrow(test_data)))
  
  x_train <- model.matrix(outcome ~ ., train_data)[,-1]
  y_train <- train_data$outcome
  
  x_test <- model.matrix(outcome ~ ., test_data)[,-1]
  
  cv_model <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
  best_lambda <- cv_model$lambda.min
  final_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)
  
  probabilities <- predict(final_model, type = "response", newx = x_test)
  result1 <- c(result1, probabilities)
}

# Combine results
dfresult_raw <- data.frame(
  lasso_gating = result1,
  truth = truth,
  group = group
)






