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
import argparse
import numpy as np
import pandas as pd
from Metalprot_learning import loader
from Metalprot_learning.train.models import FullyConnectedNet, Classifier

def distribute_tasks(path2pdbs: str, path2npzs: str, num_jobs: int, job_id: int):
    """
    Distributes pdb files across multiple cores for loading.
    """
    # print('path2pdbs =', path2pdbs, '\t', 'path2npzs =', path2npzs)
    # print('num_jobs =', num_jobs, '\t', 'job_id =', job_id)
    plist = sorted(os.listdir(path2pdbs))
    pdbs = [os.path.join(path2pdbs, path) for path in plist if '.pdb' in path]
    if path2npzs:
        zlist = sorted(os.listdir(path2npzs))
        npzs = [os.path.join(path2npzs, path[:-4] + '.npz') for path in plist
                if path[:-4] + '.npz' in zlist]
        if len(pdbs) != len(npzs):
            raise ValueError(('path2npzs is provided, but some PDB files have '
                              'missing MPNN-generated NPZ files.'))
        
        # print(len(pdbs))
        # print(pdbs)
        # print(npzs)
        # print(i % num_jobs == job_id for i in range(len(pdbs)))
        tasks = [(pdbs[i], npzs[i]) for i in range(len(pdbs)) 
                 if i % num_jobs == job_id - 1]
    else:
        tasks = [(pdbs[i], None) for i in range(len(pdbs)) 
                 if i % num_jobs == job_id - 1]
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

def run_site_enumeration(tasks: list, coordination_number: tuple, mpnn_threshold: float, required_chains: list, remove_redundant: bool):
    sources, identifiers, features = [], [], []
    failed = []
    df = pd.DataFrame(columns=['identifiers', 'source', 'distance_matrices', 
                               'encodings', 'channels', 'metal_coords', 'labels'])
    for pdb_file, npz_file in tasks:
        # try:
        for char in 'a':
            print(pdb_file)
            protein = loader.Protein(pdb_file)
            if npz_file is not None:
                with np.load(npz_file) as data:
                    mpnn_predictions = data['log_p'].reshape((-1, 21))
            else:
                mpnn_predictions = None
            fcn_cores, cnn_cores = protein.enumerate_cores(cnn=True, fcn=True, 
                                                           coordination_number=coordination_number, 
                                                           mpnn_predictions=mpnn_predictions, 
                                                           mpnn_threshold=mpnn_threshold, 
                                                           required_chains=required_chains)
            if remove_redundant:
                unique_fcn_cores, unique_cnn_cores = loader.remove_degenerate_cores(fcn_cores), loader.remove_degenerate_cores(cnn_cores)
            else:
                unique_fcn_cores, unique_cnn_cores = fcn_cores, cnn_cores
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

        # except:
        #     failed.append(pdb_file)
    return df, failed

def parse_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('path2output', type=os.path.realpath, 
                      help='Path at which to store model outputs.')
    argp.add_argument('path2pdbs', type=os.path.realpath, 
                      help='Path to directory containing input PDB files.')
    argp.add_argument('--path2npzs', type=os.path.realpath, default='',  
                      help='Path to directory containing NPZ files from '
                      'ProteinMPNN run on each PDB in path2pdbs with the '
                      '--unconditional-probs-only flag set.  If a path is '
                      'provided, residues are included during site '
                      'enumeration if their MPNN probabilities of being '
                      'D, E, H, or S exceed a threshold value.')
    argp.add_argument('--mpnn_threshold', type=float, default=0.25, 
                      help='Threshold MPNN-predicted probability of a '
                      'residue being D, E, H, or S in order for it to '
                      'be included in cores during site enumeration. '
                      '(Default: 0.25)')
    argp.add_argument('--required_chains', nargs='+', 
                      help='One-letter identifiers for the chains that '
                      'must be included in every core. If this argument '
                      'is not provided, cores with residues from any '
                      'chain are permissible.')
    argp.add_argument('--remove_redundant', action='store_true', 
                      help='If this argument is provided, redundant '
                      'cores will be removed.')
    argp.add_argument('--num_jobs', type=int, default=1, 
                      help='Number of jobs to run. (Default: 1)')
    argp.add_argument('--job_id', type=int, default=0, 
                      help='ID of current job if multiple jobs are run.')
    return argp.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path2output = args.path2output
    path2pdbs = args.path2pdbs
    path2npzs = args.path2npzs
    mpnn_threshold = args.mpnn_threshold
    required_chains = args.required_chains
    remove_redundant = args.remove_redundant
    num_jobs = args.num_jobs
    job_id = args.job_id

    COORDINATION_NUMBER = (2,4)

    tasks = distribute_tasks(path2pdbs, path2npzs, num_jobs, job_id)
    features_df, failed = run_site_enumeration(tasks, COORDINATION_NUMBER, mpnn_threshold, required_chains, remove_redundant)
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
    
    if not os.path.exists(path2output):
        os.mkdir(path2output)
    features_df.to_pickle(os.path.join(path2output, f'predictions{job_id}.pkl'))

    failed = list(filter(None, failed))
    if len(failed) > 0:
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join([line for line in failed]) + '\n')
