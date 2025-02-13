# cytoGPNet: A Tool for Accurate Clinical Outcome Prediction with Longitudinal Cytometry Data Under Limited Sample Size

## Overview
This repository provides the code for **cytoGPNet**, along with a post-hoc explanation masking algorithm. This tool implements the methodology from the paper:

> **cytoGPNet**: Enhancing Clinical Outcome Prediction Accuracy Using Longitudinal Cytometry Data in Small Cohort Studies

In short, **cytoGPNet** is designed to:
1. Efficiently handle varying cell counts per sample,
2. Incorporate time‐dependent (longitudinal) information,
3. Leverage a pre‐trained autoencoder plus Gaussian process (GP) model for robust predictions with limited subjects,
4. Offer interpretable results through a post-hoc masking algorithm that identifies important cell markers.

Below is a more comprehensive walk‐through to help new users set up and run **cytoGPNet**.

---

## Getting Started

### 1. Environment Setup
1. **Clone** (or download) this repository.
2. Install the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate cytoGPNet

## 2. Data Acquisition & Preprocessing

1. **Locate or download** the dataset(s) you plan to use. Example data and preprocessing scripts are found in the [`Data`](./Data) folder.

2. **Perform data preprocessing** (if needed) using scripts such as:
   - [`HEUvsUE_arcsinh_transform.R`](./Data/HEUvsUE_arcsinh_transform.R)
   - [`HEUvsUE_preprocess.R`](./Data/HEUvsUE_preprocess.R)

   These scripts demonstrate how to apply arcsinh transforms or any other transformations required by your dataset.

3. Make sure you **update the path and file names** in:
   - [`cytoGPNet model/pretrain.py`](./cytoGPNet%20model/pretrain.py)
   - [`cytoGPNet model/train_simplified.py`](./cytoGPNet%20model/train_simplified.py)

   so that they point to the correct preprocessed data.

