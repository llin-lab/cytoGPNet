# cytoGPNet: A Tool for Accurate Clinical Outcome Prediction with Longitudinal Cytometry Data Under Limited Sample Size

## Overview
This repository provides the code for **cytoGPNet**, along with a post-hoc explanation masking algorithm. This tool implements the methodology from the paper:

> **cytoGPNet**: Enhancing Clinical Outcome Prediction Accuracy Using Longitudinal Cytometry Data in Small Cohort Studies

In short, **cytoGPNet** is designed to:
1. Efficiently handle varying cell counts per sample,
2. Incorporate time‐dependent (longitudinal) information,
3. Leverage a pre‐trained autoencoder plus Gaussian process (GP) model for robust predictions with limited subjects,
4. Offer interpretable results through a post-hoc masking algorithm that identifies important cell markers.

Below is a more comprehensive walk‐through to set up and run **cytoGPNet**.

---

## Getting Started

### 1. Environment Setup
1. **Clone** (or download) this repository.
```
git clone https://github.com/your-username/cytoGPNet.git
cd cytoGPNet
```
3. Install the conda environment:
```
conda env create -f environment.yml
conda activate cytoGPNet
```
Note: The provided environment.yml includes platform-specific dependencies (e.g., GPU-enabled PyTorch). If you encounter compatibility issues, consider manually installing the core packages listed below instead:
- torch==1.12.1+cu113
- torchvision==0.13.1+cu113
- torchaudio==0.12.1+cu113
- gpytorch==1.13
- pandas==1.3.5
- matplotlib==3.5.3
- seaborn==0.12.0
- tqdm==4.67.1
- numpy==1.26.4

For CPU-only environments or other OS setups, please adapt the versions accordingly (e.g., use CPU builds of PyTorch).
## 2. Data Acquisition & Preprocessing

1. **Locate or download** the dataset(s) you plan to use. Example data and preprocessing scripts are found in the [`Data`](./Data) folder. All the fcs files are downloaded from **https://drive.google.com/drive/folders/1chfsOYSCsRg7kcydqyrze6B1Sg9-iInL*

2. **Perform data preprocessing** (if needed) using scripts such as:
   - [`HEUvsUE_arcsinh_transform.R`](./Data/HEUvsUE/HEUvsUE_arcsinh_transform.R)
   - [`HEUvsUE_preprocess.R`](./Data/HEUvsUE/HEUvsUE_preprocess.R)

   These scripts demonstrate how to apply arcsinh transforms or any other transformations required by your dataset.

3. Make sure you **update the path and file names** in:
   - [`cytoGPNet model/pretrain.py`](./cytoGPNet%20model/pretrain.py)
   - [`cytoGPNet model/train_simplified.py`](./cytoGPNet%20model/train_simplified.py)

   so that they point to the correct preprocessed data.

## Instructions

Below is a step‐by‐step walk‐through on how to train, test, and apply the explanation module in **cytoGPNet**.

---

### 1. Pretrain the Autoencoder
**cytoGPNet** first pre‐trains an autoencoder at the single‐cell level. This helps learn a compressed representation of the data before involving the Gaussian process.

From your command line:
```
python cytoGPNet model/pretrain.py \
    --save-dir <path/to/save/dir>
```
- --save-dir: directory to store the autoencoder checkpoint (e.g., pretrained_ae.pt) and any log files.
You can monitor the reconstruction loss in the console during training.

### 2. Train the cytoGPNet Model
After pretraining, you can fine‐tune the model end‐to‐end using the Gaussian process plus attention layers.
```
python cytoGPNet model/train_simplified.py \
    --save-dir <path/to/save/dir> \
    --pretrained-file <path/to/pretrained/>
```
- --save-dir: a directory to save the trained cytoGPNet model (e.g., model_final.pt) and logs.
- --pretrained-file: the path to your saved AE checkpoint from the pretraining step (e.g., pretrained_ae.pt).

Tips:
Keep an eye on the printed training logs (loss, accuracy, etc.) to see if your model converges.
Make sure the dataset file name in ```train_simplified.py``` matches your actual data file path.

### 3. Testing
Use ```test.py``` to evaluate performance on a held‐out test set. The script will generate predictions and save a file named ```test_result.csv```:
```
python cytoGPNet model/test.py \
    --save-dir <path/to/save/dir> \
    --trained-model <path/to/trained/model/>
```
- --trained-model: path to the final trained model (e.g., model_final.pt).
- --save-dir: specifies where the test_result.csv file is saved.
- Inspect test_result.csv for prediction probabilities, labels, or other metrics.

### 4. Explanation
The explanation script helps identify the most influential markers contributing to the model’s predictions. It produces “mask” scores (ranging from 0 to 1) for each marker, where higher scores indicate greater importance.
```
python cytoGPNet model/explanation.py \
    --model <path/to/trained/model/>
```
- --model: path to the trained cytoGPNet model.
- The script will output mask scores, which you can interpret or visualize to understand feature importance.
