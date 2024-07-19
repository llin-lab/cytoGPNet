library(tidyverse)
repo <- "~/Desktop/AutoEncoderCyTOF/HEUvsUE/" # Set your own repo where you store the fcs files
setwd(repo)
metadata <- read_csv("./train/train_labels.csv")
metadata %>% mutate(patient_id = unlist(strsplit(fcs_file, split = "[.]"))[1]) -> metadata
for (i in 1:nrow(metadata)) {
  metadata$patient_id[i] <- unlist(strsplit(metadata$fcs_file[i], split = "[.]"))[1]
}
metadata$patient_id <- as.numeric(metadata$patient_id)

metadata1 <- read_csv("./test/test_labels.csv")
metadata1 %>% mutate(patient_id = unlist(strsplit(fcs_file, split = "[.]"))[1]) -> metadata1
for (i in 1:nrow(metadata1)) {
  metadata1$patient_id[i] <- unlist(strsplit(metadata1$fcs_file[i], split = "[.]"))[1]
}
metadata1$patient_id <- as.numeric(metadata1$patient_id)

metadata_whole <- rbind(metadata, metadata1)
metadata_whole$group <- c(rep(1:5, floor(nrow(metadata_whole) / 5)), 1:(nrow(metadata_whole) %% 5))

patient_ID <- metadata_whole$patient_id


library(flowCore)
n_row <- c()
for (i in patient_ID) {
  temp <- metadata_whole %>% dplyr::filter(patient_id == i)
  file_name <- unlist(temp[1,1])
  names(file_name) <- NULL
  if (i %in% metadata$patient_id) fcs_raw <- read.flowSet(file_name, path = paste0(repo, "train/"), transformation = FALSE, package="flowCore")
  else fcs_raw <- read.flowSet(file_name, path = paste0(repo, "test/"), transformation = FALSE, package="flowCore")
  fcs_data <- exprs(fcs_raw[[1]])
  fcs_df <- as.data.frame(fcs_data)
  n_row <- c(n_row, nrow(fcs_df))

  print(i)
}


library(flowCore)
set.seed(1)

for (i in 1:nrow(metadata)) {
  file_name <- unlist(metadata[i,1])
  fcs_raw <- read.flowSet(file_name, path = paste0(repo, "train/"), transformation = FALSE, package="flowCore")
  fcs_data <- exprs(fcs_raw[[1]])
  fcs_df <- as.data.frame(fcs_data)[,3:10]
  if (nrow(fcs_df) > 1E5) fcs_df <- fcs_df[sample(1:nrow(fcs_df), 1E5),]
  
  if (file_name == metadata$fcs_file[1]) marker_label <- colnames(fcs_df)[1:8]
  else if (sum(marker_label != colnames(fcs_df)[1:8]) > 0) stop("the markers do not match across fcs files")
  fcs_value <- fcs_df[,1:8]
  fcs_value$patient <- rep(metadata$patient_id[metadata$fcs_file == file_name], nrow(fcs_value))
  fcs_value$outcome <- metadata$label[metadata$fcs_file == file_name]
  
  if (file_name == metadata$fcs_file[1]) whole_df <- fcs_value
  else whole_df <- rbind(whole_df, fcs_value)
  print(i)
}

for (i in 1:nrow(metadata1)) {
  file_name <- unlist(metadata1[i,1])
  fcs_raw <- read.flowSet(file_name, path = paste0(repo, "test/"), transformation = FALSE, package="flowCore")
  fcs_data <- exprs(fcs_raw[[1]])
  fcs_df <- as.data.frame(fcs_data)[,3:10]
  
  if (file_name == metadata1$fcs_file[1]) marker_label <- colnames(fcs_df)[1:8]
  else if (sum(marker_label != colnames(fcs_df)[1:8]) > 0) stop("the markers do not match across fcs files")
  fcs_value <- fcs_df[,1:8]
  fcs_value$patient <- rep(metadata1$patient_id[metadata1$fcs_file == file_name], nrow(fcs_value))
  fcs_value$outcome <- metadata1$label[metadata1$fcs_file == file_name]
  
  whole_df <- rbind(whole_df, fcs_value)
  print(i)
}

saveRDS(whole_df, "./whole_df_8_dims.rds")

n_row <- c()
for (i in patient_ID) {
  whole_df %>% dplyr::filter(patient == i) -> temp
  n_row <- c(n_row, nrow(temp))
}


set.seed(259)
subsample_list <- list()
for (j in 1:length(patient_ID)) {
  if (n_row[j] == 0) subsample_list[j] <- 0
  else {
    whole_df %>% dplyr::filter(patient == patient_ID[j]) -> temp
    subsample <- sample(1:nrow(temp), min(n_row[n_row > 0]))
    subsample_list[[j]] <- subsample
    if (j == 1) nk_input <- temp[subsample, ]
    else nk_input <- rbind(nk_input, temp[subsample, ])
  }
}

saveRDS(subsample_list, "./subsample_list.rds")
saveRDS(nk_input, "./HEUvsUE_8_dims_whole_input.rds") ## This nk_input is the input for the prediction models involving nk cells, with the first 5 columns (markers) as the value input, 
# Column 58 - 64 is the Demographic Data that can be ignored in CNN model if they are not applicable and Column 65 (Last column called outcome) as the prediction true label in the model.

write_csv(nk_input, "./HEUvsUE_8_dims_whole_input.csv")


set.seed(259)
subsample_list <- list()
for (j in 1:length(patient_ID)) {
  if (n_row[j] == 0) subsample_list[j] <- 0
  else {
    whole_df %>% dplyr::filter(patient == patient_ID[j]) -> temp
    subsample <- sample(1:nrow(temp), floor(min(n_row[n_row > 0]) / 2))
    subsample_list[[j]] <- subsample
    if (j == 1) nk_input <- temp[subsample, ]
    else nk_input <- rbind(nk_input, temp[subsample, ])
  }
}

saveRDS(subsample_list, "./subsample_list_50_percent.rds")
saveRDS(nk_input, "./HEUvsUE_50_percent_input.rds") ## This nk_input is the input for the prediction models involving nk cells, with the first 5 columns (markers) as the value input, 
# Column 58 - 64 is the Demographic Data that can be ignored in CNN model if they are not applicable and Column 65 (Last column called outcome) as the prediction true label in the model.

write_csv(nk_input, "./HEUvsUE_8_dims_50_percent_input.csv")



set.seed(259)
subsample_list <- list()
for (j in 1:length(patient_ID)) {
  if (n_row[j] == 0) subsample_list[j] <- 0
  else {
    whole_df %>% dplyr::filter(patient == patient_ID[j]) -> temp
    subsample <- sample(1:nrow(temp), floor(min(n_row[n_row > 0]) / 4))
    subsample_list[[j]] <- subsample
    if (j == 1) nk_input <- temp[subsample, ]
    else nk_input <- rbind(nk_input, temp[subsample, ])
  }
}

saveRDS(subsample_list, "./subsample_list_25_percent.rds")
saveRDS(nk_input, "./HEUvsUE_25_percent_input.rds") ## This nk_input is the input for the prediction models involving nk cells, with the first 5 columns (markers) as the value input, 
# Column 58 - 64 is the Demographic Data that can be ignored in CNN model if they are not applicable and Column 65 (Last column called outcome) as the prediction true label in the model.

write_csv(nk_input, "./HEUvsUE_8_dims_25_percent_input.csv")

set.seed(259)
subsample_list <- list()
for (j in 1:length(patient_ID)) {
  whole_df %>% dplyr::filter(patient == patient_ID[j]) -> temp
  subsample <- sample(1:nrow(temp), 1024)
  subsample_list[[j]] <- subsample
  if (j == 1) nk_input <- temp[subsample, ]
  else nk_input <- rbind(nk_input, temp[subsample, ])
}

saveRDS(subsample_list, "./subsample_list_1024_cells.rds")
saveRDS(nk_input, "./HEUvsUE_1024_cells_input.rds") ## This nk_input is the input for the prediction models involving nk cells, with the first 5 columns (markers) as the value input, 
# Column 58 - 64 is the Demographic Data that can be ignored in CNN model if they are not applicable and Column 65 (Last column called outcome) as the prediction true label in the model.

write_csv(nk_input, "./HEUvsUE_8_dims_1024_cells_input.csv")

metadata_whole <- rbind(metadata, metadata1)
metadata_whole$group <- c(rep(1:5, floor(nrow(metadata_whole) / 5)), 1:(nrow(metadata_whole) %% 5))

write_csv(metadata_whole, "./metadata_whole.csv")

metadata_whole %>% dplyr::filter(label == 0) -> metadata_label0
metadata_whole %>% dplyr::filter(label == 1) -> metadata_label1
set.seed(1)
metadata_label0 <- metadata_label0[sample(1:nrow(metadata_label0), floor(nrow(metadata_label0) / nrow(metadata_whole) * 40)), ]
metadata_label1 <- metadata_label1[sample(1:nrow(metadata_label1), ceiling(nrow(metadata_label1) / nrow(metadata_whole) * 40)), ]

metadata_40_patients <- rbind(metadata_label0, metadata_label1)
metadata_40_patients$group <- rep(1:5, 8)
write_csv(metadata_40_patients, "metadata_40_patients.csv")
whole_df <- readRDS("./whole_df_8_dims.rds")
n_row <- c()
for (i in patient_ID) {
  whole_df %>% dplyr::filter(patient == i) -> temp
  n_row <- c(n_row, nrow(temp))
}
names(n_row) <- as.character(patient_ID)
set.seed(259)
subsample_list <- list()
for (j in 1:nrow(metadata_40_patients)) {
  if (n_row[as.character(metadata_40_patients$patient_id[j])] == 0) subsample_list[j] <- 0
  else {
    whole_df %>% dplyr::filter(patient == metadata_40_patients$patient_id[j]) -> temp
    subsample <- sample(1:nrow(temp), min(n_row[n_row > 0]))
    subsample_list[[j]] <- subsample
    if (j == 1) nk_input <- temp[subsample, ]
    else nk_input <- rbind(nk_input, temp[subsample, ])
  }
}
saveRDS(subsample_list, "./subsample_list_40_patients.rds")
saveRDS(nk_input, "./HEUvsUE_40_patients_input.rds") ## This nk_input is the input for the prediction models involving nk cells, with the first 5 columns (markers) as the value input, 
# Column 58 - 64 is the Demographic Data that can be ignored in CNN model if they are not applicable and Column 65 (Last column called outcome) as the prediction true label in the model.

write_csv(nk_input, "./HEUvsUE_8_dims_40_patients_input.csv")

nk_input <- readRDS("./HEUvsUE_8_dims_whole_input.rds")
nk_input[,1:8] <- asinh(nk_input[,1:8]/5)
write_csv(nk_input, "./HEUvsUE_8_dims_whole_arcsinh_input.csv")


nk_input <- readRDS("./HEUvsUE_40_patients_input.rds")
nk_input[,1:8] <- asinh(nk_input[,1:8]/5)
write_csv(nk_input, "./HEUvsUE_8_dims_40_patients_arcsinh_input.csv")



