#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # Number of CPU to request for the job
#SBATCH --mem=12GB                  # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=00-02:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL        # When should you receive an email?
#SBATCH --output=logs/%u.%j.out     # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory
#SBATCH --constraint="skylake"      # Constrain to non-preemtable Nvidia Tesla V100 Skylake GPUs

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=student                         # The partition you've been assigned
#SBATCH --account=student                           # The account you've been assigned (normally student)
#SBATCH --qos=studentqos                            # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=nc.voigt.2024@mitb.smu.edu.sg   # Who should receive the email notifications
#SBATCH --job-name=Capstone_MVP_Test                # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the enviromnent, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Python/3.13.1-GCCcore-13.3.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# Create a virtual environment can be commented off if you already have a virtual environment
python3 -m venv ~/Capstone

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/Capstone/bin/activate

# Find out which GPU you are using
srun whichgpu

# If you require any packages, install it as usual before the srun job submission.
pip3 install -q numpy
pip3 install -q scipy
pip3 install -q torch torchvision torchaudio
pip3 install -q mamba_ssm

# Submit your job to the cluster
srun --gres=gpu:1 python run.py --graph_size 20 --tour_heuristic random --tour_method greedy --n_epochs 20 --no_progress_bar
