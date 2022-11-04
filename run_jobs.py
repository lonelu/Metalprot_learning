#!/usr/bin/env python3

'''
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

Wrapper script to run jobs for Metalprot_learning on wynton. Code 
adapted from Xingjie Pan's loop-helix-loop reshaping repository.

Usage:
    ./run_jobs.py <job-name> <job-script> [options]

Arguments:
    <job-name> 
        Job name of user's choosing.

    <job-script>
        Job script for the analysis.

Options:
    --job-distributor=<JD>, -d=<JD>  [default: sequential]
        Job distributor that runs the jobs.

    --num-jobs=<NJ>, -n=<NJ>
        Number of jobs for parallel run.

    --time=<T>, -t=<T>  [default: 05:00:00]
        Time to alot for each parallel run. For example, 05:00:00 will alot 5 hours.

    --processing-unit=<P>, -p=<P>  [default: cpu]
        Defines the type of processing unit to run job on. Y designates GPU usage.

    --job-arguments=<A>, -a=<A>
        String of additional command line arguments to the job script.
'''

import docopt
import shutil
import os
import subprocess

def run_SGE(job_name: str, num_jobs: int, path: str, job_script: str, job_arguments: str, time: str, processing_unit: str, mem_free_GB=3, scratch_space_GB=1, keep_job_output_path=True):

    """Runs SGE job on UCSF Wynton cluster.

    Args:
        num_jobs (int): Number of jobs.
        path (str): Path to output directory to store job logs and output files.
        job_script (str): Path to the script to be submitted.
        job_arguments (str, optional): Command line arguments to the job script.
        time (str, optional): Time alloted for each task.
        job_arguments (str, optional): Command line arguments to the job script.
        gpu (str): Defines the processing unit to run job on.
        mem_free_GB (int, optional): Defaults to 3.
        scratch_space_GB (int, optional): Defaults to 1.
        keep_job_output_path (bool, optional): Defaults to True.
    """

    job_output_path = os.path.join(path, "job_outputs")
    if not keep_job_output_path and os.path.exists(job_output_path): 
        shutil.rmtree(job_output_path)

    if not os.path.exists(job_output_path):
        os.makedirs(job_output_path)

    qsub_command = ['qsub',
                        '-cwd'] \
                     + ['-N', job_name,
                        '-t', '1-{0}'.format(num_jobs),
                        '-l', 'h_rt={0}'.format(time),
                        '-l', 'hostname=qb3-id*', # ensure AVX2 instructions
                        '-o', job_output_path,
                        '-e', job_output_path,
                        './job_scripts/activate_env.sh',
                        job_script] \
                        + job_arguments.split() \
                        + [' --num_jobs ', str(num_jobs), ' --job_id ']

    if processing_unit == 'gpu':
        append = ['-q', 'gpu.q', '-pe', 'smp', num_jobs]
        qsub_command[1:1] = append

    subprocess.run(qsub_command)

def run_sequential(job_name: str, path: str, job_script: str, job_arguments: str, keep_job_output_path=True):
    command = [job_script] + job_arguments.split()
    subprocess.run(command)

if __name__ == '__main__':
    
    arguments = docopt.docopt(__doc__)
    job_script = arguments['<job-script>']
    job_name = arguments['<job-name>']

    path = arguments['--job-arguments'].split()[0]

    if arguments['--job-distributor'] == 'SGE':
        num_jobs = arguments['--num-jobs'] if arguments['--num-jobs'] else '1'
        run_SGE(job_name, num_jobs, path, job_script, arguments['--job-arguments'], arguments['--time'], arguments['--processing-unit'])

    elif arguments['--job-distributor'] == 'sequential':
        run_sequential(job_name, path, job_script, arguments['--job-arguments'])

    else:
        raise IOError('Unknown job distributor: {0}'.format(arguments['--job-distributor']))
