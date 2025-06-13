library(tidyverse)
library(data.table)  # Add data.table library

repo <- "~/Desktop/AutoEncoderCyTOF/HEUvsUE/" # Set your own repo where you store the fcs files
setwd(repo)

# Read and process metadata
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
  if (i %in% metadata$patient_id) 
    fcs_raw <- read.flowSet(file_name, path = paste0(repo, "/train"), transformation = FALSE, package="flowCore")
  else 
    fcs_raw <- read.flowSet(file_name, path = paste0(repo, "/test"), transformation = FALSE, package="flowCore")
  fcs_data <- exprs(fcs_raw[[1]])
  fcs_df <- as.data.frame(fcs_data)
  n_row <- c(n_row, nrow(fcs_df))
  print(i)
}

library(flowCore)
set.seed(1)
for (i in 1:nrow(metadata)) {
  file_name <- unlist(metadata[i,1])
  fcs_raw <- read.flowSet(file_name, path = paste0(repo, "/train"), transformation = FALSE, package="flowCore")
  fcs_data <- exprs(fcs_raw[[1]])
  fcs_df <- as.data.frame(fcs_data)[,3:10]
  if (nrow(fcs_df) > 1E5) fcs_df <- fcs_df[sample(1:nrow(fcs_df), 1E5),]
  
  if (file_name == metadata$fcs_file[1]) marker_label <- colnames(fcs_df)[1:8]
  else if (sum(marker_label != colnames(fcs_df)[1:8]) > 0) stop("the markers do not match across fcs files")
  fcs_value <- fcs_df[,1:8]
  fcs_value$patient_id <- rep(metadata$patient_id[metadata$fcs_file == file_name], nrow(fcs_value))
  fcs_value$outcome <- metadata$label[metadata$fcs_file == file_name]
  
  if (file_name == metadata$fcs_file[1]) whole_df <- fcs_value
  else whole_df <- rbind(whole_df, fcs_value)
  print(i)
}

for (i in 1:nrow(metadata1)) {
  file_name <- unlist(metadata1[i,1])
  fcs_raw <- read.flowSet(file_name, path = paste0(repo, "/test"), transformation = FALSE, package="flowCore")
  fcs_data <- exprs(fcs_raw[[1]])
  fcs_df <- as.data.frame(fcs_data)[,3:10]
  
  if (file_name == metadata1$fcs_file[1]) marker_label <- colnames(fcs_df)[1:8]
  else if (sum(marker_label != colnames(fcs_df)[1:8]) > 0) stop("the markers do not match across fcs files")
  fcs_value <- fcs_df[,1:8]
  fcs_value$patient_id <- rep(metadata1$patient_id[metadata1$fcs_file == file_name], nrow(fcs_value))
  fcs_value$outcome <- metadata1$label[metadata1$fcs_file == file_name]
  
  whole_df <- rbind(whole_df, fcs_value)
  print(i)
}

saveRDS(whole_df, "./whole_df_8_dims.rds")

# Convert to data.table for more efficient operations
whole_df_dt <- as.data.table(whole_df)
setkey(whole_df_dt, patient_id)  # Index for faster filtering

# Get counts per patient (more efficient)
patient_counts <- whole_df_dt[, .N, by = patient_id]
n_row_counts <- patient_counts$N
names(n_row_counts) <- patient_counts$patient_id

# Create patient ID to index mapping for faster lookups
patient_id_to_idx <- setNames(1:length(patient_ID), as.character(patient_ID))

# Modified process for subsampling - process in smaller batches
set.seed(259)
subsample_list <- list()
nk_input <- NULL
batch_size <- 5  # Process 5 patients at a time
num_batches <- ceiling(length(patient_ID)/batch_size)

for (batch in 1:num_batches) {
  start_idx <- (batch-1)*batch_size + 1
  end_idx <- min(batch*batch_size, length(patient_ID))
  
  for (j in start_idx:end_idx) {
    pat_id <- patient_ID[j]
    pat_id_char <- as.character(pat_id)
    
    if (n_row_counts[pat_id_char] == 0 || is.na(n_row_counts[pat_id_char])) {
      subsample_list[j] <- 0
    } else {
      # Use data.table for efficient filtering
      temp <- whole_df_dt[patient == pat_id]
      min_rows <- min(n_row_counts[n_row_counts > 0])
      row_count <- nrow(temp)
      
      if (row_count > 0) {
        sample_size <- min(row_count, min_rows)
        subsample <- sample(1:row_count, sample_size)
        subsample_list[[j]] <- subsample
        
        if (is.null(nk_input)) {
          nk_input <- temp[subsample, ]
        } else {
          nk_input <- rbind(nk_input, temp[subsample, ])
        }
      }
    }
  }
  
  # Clear memory between batches
  gc()
}

saveRDS(subsample_list, "./subsample_list.rds")
saveRDS(nk_input, "./HEUvsUE_8_dims_whole_input.rds")
write_csv(nk_input, "./HEUvsUE_8_dims_whole_input.csv")

# For 50% sampling
set.seed(259)
subsample_list <- list()
nk_input <- NULL
min_rows <- min(n_row_counts[n_row_counts > 0])
half_min_rows <- floor(min_rows / 2)

for (batch in 1:num_batches) {
  start_idx <- (batch-1)*batch_size + 1
  end_idx <- min(batch*batch_size, length(patient_ID))
  
  for (j in start_idx:end_idx) {
    pat_id <- patient_ID[j]
    pat_id_char <- as.character(pat_id)
    
    if (n_row_counts[pat_id_char] == 0 || is.na(n_row_counts[pat_id_char])) {
      subsample_list[j] <- 0
    } else {
      temp <- whole_df_dt[patient == pat_id]
      row_count <- nrow(temp)
      
      if (row_count > 0) {
        sample_size <- min(row_count, half_min_rows)
        subsample <- sample(1:row_count, sample_size)
        subsample_list[[j]] <- subsample
        
        if (is.null(nk_input)) {
          nk_input <- temp[subsample, ]
        } else {
          nk_input <- rbind(nk_input, temp[subsample, ])
        }
      }
    }
  }
  
  gc()
}

saveRDS(subsample_list, "./subsample_list_50_percent.rds")
saveRDS(nk_input, "./HEUvsUE_50_percent_input.rds")
write_csv(nk_input, "./HEUvsUE_8_dims_50_percent_input.csv")

# For 25% sampling
set.seed(259)
subsample_list <- list()
nk_input <- NULL
quarter_min_rows <- floor(min_rows / 4)

for (batch in 1:num_batches) {
  start_idx <- (batch-1)*batch_size + 1
  end_idx <- min(batch*batch_size, length(patient_ID))
  
  for (j in start_idx:end_idx) {
    pat_id <- patient_ID[j]
    pat_id_char <- as.character(pat_id)
    
    if (n_row_counts[pat_id_char] == 0 || is.na(n_row_counts[pat_id_char])) {
      subsample_list[j] <- 0
    } else {
      temp <- whole_df_dt[patient == pat_id]
      row_count <- nrow(temp)
      
      if (row_count > 0) {
        sample_size <- min(row_count, quarter_min_rows)
        subsample <- sample(1:row_count, sample_size)
        subsample_list[[j]] <- subsample
        
        if (is.null(nk_input)) {
          nk_input <- temp[subsample, ]
        } else {
          nk_input <- rbind(nk_input, temp[subsample, ])
        }
      }
    }
  }
  
  gc()
}

saveRDS(subsample_list, "./subsample_list_25_percent.rds")
saveRDS(nk_input, "./HEUvsUE_25_percent_input.rds")
write_csv(nk_input, "./HEUvsUE_8_dims_25_percent_input.csv")

# For 1024 cells sampling
set.seed(259)
subsample_list <- list()
nk_input <- NULL

for (batch in 1:num_batches) {
  start_idx <- (batch-1)*batch_size + 1
  end_idx <- min(batch*batch_size, length(patient_ID))
  
  for (j in start_idx:end_idx) {
    pat_id <- patient_ID[j]
    
    temp <- whole_df_dt[patient == pat_id]
    row_count <- nrow(temp)
    
    if (row_count > 0) {
      sample_size <- min(row_count, 1024)
      subsample <- sample(1:row_count, sample_size)
      subsample_list[[j]] <- subsample
      
      if (is.null(nk_input)) {
        nk_input <- temp[subsample, ]
      } else {
        nk_input <- rbind(nk_input, temp[subsample, ])
      }
    }
  }
  
  gc()
}

saveRDS(subsample_list, "./subsample_list_1024_cells.rds")
saveRDS(nk_input, "./HEUvsUE_1024_cells_input.rds")
write_csv(nk_input, "./HEUvsUE_8_dims_1024_cells_input.csv")

# Process metadata for 40 patients
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

# For 40 patients sampling
set.seed(259)
subsample_list <- list()
nk_input <- NULL
min_rows <- min(n_row_counts[n_row_counts > 0])

for (j in 1:nrow(metadata_40_patients)) {
  pat_id <- metadata_40_patients$patient_id[j]
  pat_id_char <- as.character(pat_id)
  
  if (!pat_id_char %in% names(n_row_counts) || n_row_counts[pat_id_char] == 0) {
    subsample_list[j] <- 0
  } else {
    temp <- whole_df_dt[patient == pat_id]
    row_count <- nrow(temp)
    
    if (row_count > 0) {
      sample_size <- min(row_count, min_rows)
      subsample <- sample(1:row_count, sample_size)
      subsample_list[[j]] <- subsample
      
      if (j == 1) {
        nk_input <- temp[subsample, ]
      } else {
        nk_input <- rbind(nk_input, temp[subsample, ])
      }
    }
  }
  
  # Clear memory more frequently
  if (j %% 5 == 0) gc()
}

saveRDS(subsample_list, "./subsample_list_40_patients.rds")
saveRDS(nk_input, "./HEUvsUE_40_patients_input.rds")
write_csv(nk_input, "./HEUvsUE_8_dims_40_patients_input.csv")

# Transform with arcsinh
nk_input <- readRDS("./HEUvsUE_8_dims_whole_input.rds")
nk_input[,1:8] <- asinh(nk_input[,1:8]/5)
write_csv(nk_input, "./HEUvsUE_8_dims_whole_arcsinh_input.csv")

nk_input <- readRDS("./HEUvsUE_40_patients_input.rds")
nk_input[,1:8] <- asinh(nk_input[,1:8]/5)
write_csv(nk_input, "./HEUvsUE_8_dims_40_patients_arcsinh_input.csv")
