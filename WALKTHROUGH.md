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

### 2. Data Acquisition & Preprocessing
1. Locate or download the dataset(s) you plan to use. Example data and preprocessing scripts are in the Data folder.
