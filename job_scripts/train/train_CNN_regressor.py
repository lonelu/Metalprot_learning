#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This script runs CNN model training. 
"""

#imports
from Metalprot_learning.train.train import train_model
import os
import sys
import datetime

def distribute_tasks(MODELS: list):
    """Distributes batch jobs accross mutliple cores.

    Args:
        no_jobs (int): Number of jobs.
        job_id (int): Job id.
        path2models (str): Path to directory containing model subdirectories.

    Returns:
        tasks (list): List of models to be train by task.
    """

    path2output = sys.argv[1] #note that this is not actually where output files are written to. they are written to the model directories.
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1

    tasks = [MODELS[i] for i in range(0,len(MODELS)) if i % no_jobs == job_id]
    return path2output, tasks

if __name__ == '__main__':
    PATH2FEATURES = '/home/gpu/jzhang1198/data/ZN_binding_cores/cnn-cores-2022-07-31/compiled_data.pkl'
    MODELS = [
        {'seed': 69, 
        'batch_size': 14, 
        'lr': 0.00020894227630869483, 
        'encodings': True, 
        'epochs': 200, 
        'block_n1': {'in': 40, 'out': 8, 'padding': 1, 'dropout_n1': 0.18632645255981992}, 
        'block0': {'out': 64, 'kernel_size': 3, 'padding': 1, 'dropout_0': 0.6173822866461347}, 
        'block1': {'dilation_residual': 1, 'out': 128, 'kernel_size': 1, 'padding': 0, 'kernel_size_pool': 2, 'dropout_1': 0.3033101227313487}, 
        'block2': {'dilation_residual': 1, 'out': 256, 'kernel_size': 1, 'padding': 0, 'kernel_size_pool': 2, 'dropout_2': 0.311068933631938}, 
        'block3': {'dilation_residual': 1, 'dropout_3': 0.30536913307953484}, 
        'linear1': {'out': 512, 'dropout_l1': 0.3}, 
        'linear2': {'out': 256, 'dropout_l2': 0.3}, 
        'linear3': {'out': 48}}
    ]

    path2output, tasks = distribute_tasks(MODELS)
    for model in tasks:
        today = datetime.datetime.now()
        dirname = os.path.join(path2output, '-'.join(str(i) for i in [today.day, today.month, today.year, model['seed']]))
        os.mkdir(dirname)
        train_model(dirname, model, PATH2FEATURES, 'CNN')
        print(f'Execution time: {(datetime.datetime.now() - today)/ (60**2)} hours')