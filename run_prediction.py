#!/usr/bin/env python3
"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>
Script for running metal binding site prediction pipeline.
"""

#imports
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from Metalprot_learning import loader
from Metalprot_learning.train.models import FullyConnectedNet, Classifier

def distribute_tasks(path2pdbs: str, no_jobs: int, job_id: int):
    """
    Distributes pdb files across multiple cores for loading.
    """
    pdbs = [os.path.join(path2pdbs, file) for file in os.listdir(path2pdbs) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]
    return tasks

def instantiate_models():
    classifier = Classifier()
    classifier.load_state_dict(torch.load('./trained_models/classifier/best_model9.pth', map_location='cpu')['state_dict'])
    classifier.eval()
    classifier = classifier.double()

    with open('./trained_models/regressor/config.json', 'r') as f:
        config = json.load(f)
    regressor = FullyConnectedNet(
        config['input'], config['l1'], config['l2'], config['l3'], 
        config['output'], config['input_dropout'], config['hidden_dropout'])
    regressor.load_state_dict(torch.load('./trained_models/regressor/model.pth', map_location='cpu'))
    regressor.eval()
    return classifier, regressor

def run_site_enumeration(tasks: list, coordination_number: tuple):
    sources, identifiers, features = [], [], []
    failed = []
    df = pd.DataFrame()
    for pdb_file in tasks:
        try:
            print(pdb_file)
            protein = loader.MetalloProtein(pdb_file)
            fcn_cores, cnn_cores = protein.enumerate_cores(cnn=True, fcn=True, coordination_number=coordination_number)
            unique_fcn_cores, unique_cnn_cores = loader.remove_degenerate_cores(fcn_cores), loader.remove_degenerate_cores(cnn_cores)
            identifiers, distance_matrices, encodings, channels, metal_coordinates, labels = [], [], [], [], [], []
            for fcn_core, cnn_core in zip(unique_fcn_cores, unique_cnn_cores):
                identifiers.append(fcn_core.identifiers)
                distance_matrices.append(fcn_core.distance_matrix)
                encodings.append(fcn_core.encoding)
                channels.append(cnn_core.channels)
                metal_coordinates.append(fcn_core.metal_coords)
                labels.append(fcn_core.label)
            df = pd.concat([df, pd.DataFrame(
                {'identifiers': identifiers, 
                'source': [pdb_file] * len(identifiers),
                'distance_matrices': distance_matrices,
                'encodings': encodings,
                'channels': channels,
                'metal_coords': metal_coordinates,
                'labels': labels})])

        except:
            failed.append(pdb_file)
    return df, failed

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    PATH2PDBS = '/Users/jonathanzhang/Documents/ucsf/degrado/DeGrado-Lab-Notebook/metal-binding/experiments/20221002_edge_case_testing/data/edge_cases'
    COORDINATION_NUMBER = (2,4)

    tasks = distribute_tasks(PATH2PDBS, no_jobs, job_id)
    features_df, failed = run_site_enumeration(tasks, COORDINATION_NUMBER)
    classifier_features, regressor_features = np.stack(features_df['channels'].tolist(), axis=0), np.hstack([np.vstack([matrix.flatten() for matrix in features_df['distance_matrices'].tolist()]), np.vstack(features_df['encodings'])])
    classifier, regressor = instantiate_models()
    classifications = classifier.forward(torch.from_numpy(classifier_features)).cpu().detach().numpy()
    rounded_classifications = classifications.round()
    metal_site_inds = np.argwhere(classifications == 1).flatten()
    _regressions = regressor.forward(torch.from_numpy(regressor_features[metal_site_inds])).cpu().detach().numpy().round()
    
    regressions = np.zeros((len(classifications), 48))
    regressions[metal_site_inds] = _regressions

    features_df['classifications'] = classifications
    features_df['rounded_classifications'] = rounded_classifications
    features_df['regressions'] = list(regressions)
    features_df.to_pickle(os.path.join(path2output, f'predictions{job_id}.pkl'))

    failed = list(filter(None, failed))
    if len(failed) > 0:
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join([line for line in failed]) + '\n')