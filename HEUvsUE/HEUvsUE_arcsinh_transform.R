
# Feb 20 Updated
library(tidyverse)
repo <- "~/Desktop/AutoEncoderCyTOF/HEUvsUE/" # Set your own repo where you store the csv files
setwd(repo)

HEUvsUE_input <- read_csv("./HEUvsUE_8_dims_whole_input.csv")
HEUvsUE_input[,1:8] <- asinh(HEUvsUE_input[,1:8]/5)
write_csv(HEUvsUE_input, "./HEUvsUE_8_dims_whole_arcsinh_input.csv")

HEUvsUE_input <- read_csv("./HEUvsUE_8_dims_50_percent_input.csv")
HEUvsUE_input[,1:8] <- asinh(HEUvsUE_input[,1:8]/5)
write_csv(HEUvsUE_input, "./HEUvsUE_8_dims_50_percent_arcsinh_input.csv")

HEUvsUE_input <- read_csv("./HEUvsUE_8_dims_25_percent_input.csv")
HEUvsUE_input[,1:8] <- asinh(HEUvsUE_input[,1:8]/5)
write_csv(HEUvsUE_input, "./HEUvsUE_8_dims_25_percent_arcsinh_input.csv")

HEUvsUE_input <- read_csv("./HEUvsUE_8_dims_40_patients_input.csv")
HEUvsUE_input[,1:8] <- asinh(HEUvsUE_input[,1:8]/5)
write_csv(HEUvsUE_input, "./HEUvsUE_8_dims_40_patients_arcsinh_input.csv")


# Feb 21 Updated

library(tidyverse)
repo <- "~/Desktop/AutoEncoderCyTOF/HEUvsUE/" # Set your own repo where you store the csv files
setwd(repo)

HEUvsUE_input <- read_csv("./HEUvsUE_8_dims_1024_cells_input.csv")
HEUvsUE_input[,1:8] <- asinh(HEUvsUE_input[,1:8]/5)
write_csv(HEUvsUE_input, "./HEUvsUE_8_dims_1024_cells_arcsinh_input.csv")
