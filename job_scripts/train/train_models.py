#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs model training. 
"""

#imports
from Metalprot_learning.trainer.model_trainer import tune_model
from ray import tune
import sys
import os
import json

def distribute_tasks(no_jobs: int, job_id: int, path2models: str):
    """Distributes batch jobs accross mutliple cores.

    Args:
        no_jobs (int): Number of jobs.
        job_id (int): Job id.
        path2models (str): Path to directory containing model subdirectories.

    Returns:
        tasks (list): List of models to be train by task.
    """

    models = [os.path.join(path2models, i) for i in os.listdir(path2models)]
    tasks = [models[i] for i in range(0,len(models)) if i % no_jobs == job_id]
    return tasks

def run_train(task: str, feature_file: str):

    try:
        with open(os.path.join(task, 'hyperparams.json'), 'r') as f:
            hyperparams = json.load(f)

    except:
        print(f'No hyperparams.json file found in {task}')
        return

    arch = hyperparams['arch']
    combination = (hyperparams['epochs'], hyperparams['batch_size'], hyperparams['lr'], hyperparams['loss_fn'], hyperparams['optimizer'])
    partition = hyperparams['partition']
    seed = hyperparams['seed']
    train_model(task, arch, feature_file, partition, seed, combination)

if __name__ == '__main__':
    path2output = sys.argv[1] #note that this is not actually where output files are written to. they are written to the model directories.
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    #provide paths to observations and labels
    path2features = '/home/gpu/jzhang1198/data/ZN_binding_cores/datasetV2/compiled_features.pkl'
    coordinating_resis = 4
    no_neighbors = 1
    no_samples = 100
    seed = 42

    config = {'l1': [],
        'l2': [],
        'batch_size': [],
        'lr': [],
        'epochs': []}

    tune_model(path2output, seed, no_samples, config, coordinating_resis, no_neighbors, path2features)
    
