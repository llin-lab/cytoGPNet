#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 02:02:09 2022

@author: jz259
"""

import pickle
import pandas as pd
import numpy as np
from numpy.random import seed; seed(111)
import torch
from torch.utils.data import Dataset
import os


class CyTOF_Dataset(Dataset):
    # Data must be load into the form of [batch_size (samples), channels (= 1), height = (cells), width (markers)]
    def __init__(self, datadir, name, mode='train'):
        self.datadir = datadir
        self.name = name
        self.mode = mode
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[1][idx]
        #cluster = self.labels[idx]
        return {'tensor': torch.from_numpy(sample).float()}

    def _load_data(self):
        allData = pickle.load(open(os.path.join(self.datadir, self.name), "rb" ))
        metaData = allData["cytof_files"]
        cytoData = allData["expr_list"]
        n_sample, n_cell, n_marker, n_channel = cytoData.shape
        #cytoData = cytoData.reshape((n_sample, n_channel, n_cell, n_marker))
        markerNames = allData["marker_names"]

        # inspect the data
        print("\nFirst 5 rows of metaData: ")
        print(metaData.head(),"\n")

        print("Dimensions of cytoData: ",cytoData.shape,"\n")
        print("Names of the 27 makers: \n",markerNames.values)
        #data = labels.merge(data, left_index=True, right_index=True)

        
        return metaData, cytoData, markerNames