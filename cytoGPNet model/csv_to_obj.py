#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:02:12 2025

@author: jz259
"""

import pandas as pd
import numpy as np
import pickle

import os
import argparse


def setup_args():

    options = argparse.ArgumentParser()

    # data directory
    options.add_argument('-datadir', '--data-dir', action="store", dest="data_dir", default='./HEUvsUE')
    options.add_argument('--csv-file', action="store", dest="csv_file", default='HEUvsUE_8_dims_whole_arcsinh_input.csv')
    options.add_argument('--metadata-file', action="store", dest="metadata_file", default='metadata_whole.csv')
    options.add_argument('--num-markers', action="store", dest="num_markers", default=8, type=int)
    options.add_argument('-fold', action="store", dest="fold", default=1, type=int)
    
    return options.parse_args()


args = setup_args()


def csv_to_obj(csv_file, metadata_file, obj_file, num_markers=19, group_filter=None):
    # Load metadata and filter by group
    cytof_files = pd.read_csv(metadata_file)
    if group_filter is not None:
        cytof_files = cytof_files[cytof_files['group'].isin(group_filter)]
    selected_patients = cytof_files['patient_id'].unique()

    # Load single-cell data and filter for selected patients
    df = pd.read_csv(csv_file)
    df = df[df['patient_id'].isin(selected_patients)]
    
    marker_cols = df.columns[:num_markers]
    has_treatment = 'Treatment' in df.columns
    group_cols = ['patient_id', 'Treatment'] if has_treatment else ['patient_id']

    # Group the data and determine max number of cells
    grouped = df.groupby(group_cols)
    max_cells = max(grouped.size())

    # Prepare data list
    expr_list = []
    sample_ids = []
    
    for name, group in grouped:
        # Extract and pad marker expressions
        expr = group[marker_cols].to_numpy()
        n_cells = expr.shape[0]
        if n_cells < max_cells:
            padding = np.zeros((max_cells - n_cells, num_markers))
            expr = np.vstack([expr, padding])
        expr_list.append(expr)

        # Record sample info
        if has_treatment:
            sample_ids.append({'patient_id': name[0], 'Treatment': name[1]})
        else:
            sample_ids.append({'patient_id': name})

    # Convert to final shape: [n_samples, n_channels, n_cells, n_markers]
    expr_array = np.array(expr_list)
    if has_treatment:
        df_info = pd.DataFrame(sample_ids)
        unique_patients = df_info['patient_id'].unique()
        unique_treatments = df_info['Treatment'].unique()
        expr_reshaped = np.zeros((len(unique_patients), len(unique_treatments), max_cells, num_markers))

        for i, pid in enumerate(unique_patients):
            for j, trt in enumerate(unique_treatments):
                match_idx = ((df_info['patient_id'] == pid) & (df_info['Treatment'] == trt)).values
                if any(match_idx):
                    expr_reshaped[i, j] = expr_array[np.where(match_idx)[0][0]]
    else:
        expr_reshaped = expr_array[:, np.newaxis, :, :]  # [n_samples, 1, n_cells, n_markers]

    # Save everything
    AllData = {
        'expr_list': expr_reshaped.astype(np.float32),
        'cytof_files': cytof_files.reset_index(drop=True),
        'marker_names': list(marker_cols)
    }

    with open(obj_file, 'wb') as f:
        pickle.dump(AllData, f)

    print(f"Filtered by group {group_filter}. Dataset saved to {obj_file} with shape {expr_reshaped.shape}")

# Example usage
csv_file = os.path.join(args.data_dir, args.csv_file)
metadata_file = os.path.join(args.data_dir, args.metadata_file)

os.makedirs(os.path.join(args.data_dir, f"fold{args.fold}"), exist_ok=True)

# Create training and testing data for each fold number as input argument
train_obj_file = os.path.join(args.data_dir, f"fold{args.fold}", f"train_Data.obj")
folds = [1, 2, 3, 4, 5]
folds.remove(args.fold)
csv_to_obj(csv_file = csv_file, metadata_file = metadata_file, obj_file = train_obj_file, num_markers = args.num_markers, group_filter = folds)


test_obj_file = os.path.join(args.data_dir, f"fold{args.fold}", f"test_Data.obj")
csv_to_obj(csv_file = csv_file, metadata_file = metadata_file, obj_file = test_obj_file, num_markers = args.num_markers, group_filter = [args.fold])






