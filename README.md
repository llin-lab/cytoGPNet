# cytoGPNet: A tool for accurate clinical outcome prediction with longitudinal cytometry data under limited sample size
## Overview: 

This repository contains the codes for cytoGPNet and perform post-hoc explanation masking algorithm. This tool is the result of the publication entitled "cytoGPNet:Enhancing Clinical Outcome Prediction Accuracy with Longitudinal Cytometry Data under Limited Sample Size".

## Getting Started:

### Depandencies
This package uses example data listed in the paper and the pre-processing procedure codes can be found in each data folder. The environment file for our model can be installed using the code below.
```
conda env create -f environment.yml
conda activate cytoGPNet
```

### Instructions
#### Training
* Download dataset and perform data pre-processing.
* Verify the dataset file name in the ``cytoGPNet model/pretrain.py`` and ``cytoGPNet model/train_simplified.py``
* Pretrain autoencoder using the code below:
```
python cytoGPNet\ model/pretrain.py --save-dir <path/to/save/dir>
```
* Start training cytoGPNet model:
```
python cytoGPNet\ model/train_simplified.py --save-dir <path/to/save/dir> --pretrained-file <path/to/pretrained/>
```
#### Testing
* The testing code is contained in ``cytoGPNet model/test.py`` and the prediction score for testing data will be saved in ``test_result.csv`` under specified folder using the code below:
```
python cytoGPNet\ model/test.py --save-dir <path/to/save/dir> --trained-model <path/to/trained/model/>
```


#### Explanation
* Explanation algorithm code is contained in ``cytoGPNet model/explanation.py`` and the mask score can be computed through this file.
