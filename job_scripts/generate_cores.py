#!/usr/bin/env python3

"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Note that this script is meant to be called from within a SGE bash submission script. 
"""

#imports 
import os
import sys
from Metalprot_learning.core_generator import construct_training_example
from Metalprot_learning import utils

def distribute_tasks(path2examples: str, no_jobs: int, job_id: int):
    """Distributes pdb files for core generation.

    Args:
        path2examples (str): Path to directory containing examples.
        no_jobs (int): Total number of jobs.
        job_id (int): The job id.

    Returns:
        tasks (list): list of pdb files assigned to particular job id.
    """
    pdbs = [os.path.join(path2examples, file) for file in os.listdir(path2examples) if '.pdb' in file]
    tasks = [pdbs[i] for i in range(0, len(pdbs)) if i % no_jobs == job_id]
    return tasks

def run_construct_training_example(file: str, path2output: str):
    """Calls main function from get_binding_cores.py to construct training example. For quality control and de-bugging purposes, there are multiple try/except statements.

    Args:
        file (str): Path to pdb file.
        path2output (str): Path to output.

    Returns:
        failed_file_line: Line to write to failed.txt
    """

    try:
        construct_training_example(file,path2output)
        failed_file_line = None

    except utils.NoCoresError as e:
        failed_file_line = file + ' No cores identified'

    except utils.DistMatDimError as e:
        failed_file_line = file + ' Distance matrix dimensions are incorrect'

    except utils.LabelDimError as e:
        failed_file_line = file + ' Label dimensions are incorrect'

    except utils.EncodingDimError as e:
        failed_file_line = file + ' Encoding dimensions are incorrect'

    except utils.EncodingError as e:
        failed_file_line = file + ' Unrecognized amino acid during sequence encoding'

    except utils.PermutationError as e:
        failed_file_line = file + ' Issue with permutation of fragments'

    except:
        failed_file_line = file + ' Unknown error occured'

    return failed_file_line

if __name__ == '__main__':
    path2output = sys.argv[1] #path to store outputs    
    no_jobs = 1
    job_id = 0

    if len(sys.argv) > 3:
        no_jobs = int(sys.argv[2])
        job_id = int(sys.argv[3]) - 1
    
    #YOU WILL NEED TO EDIT THIS PATH HERE
    path2examples = '/wynton/home/rotation/jzhang1198/data/metalprot_learning/ZN_binding_cores/src'

    failed = []
    tasks = distribute_tasks(path2examples, no_jobs, job_id)
    for file in tasks:
        failed_file_line = run_construct_training_example(file, path2output)
        failed.append(failed_file_line)

    failed.remove('None')
    if len(failed) > 0:
        with open(os.path.join(path2output, 'failed.txt'), 'a') as f:
            f.write('\n'.join([line for line in failed]))