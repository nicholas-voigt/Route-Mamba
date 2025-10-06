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
#SBATCH --constraint="48gb"

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
# python3 -m venv ~/Capstone

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
source ~/Capstone/bin/activate

# Find out which GPU you are using
srun whichgpu

# If you require any packages, install it as usual before the srun job submission.
pip3 install -q --no-cache-dir numpy
pip3 install -q --no-cache-dir scipy
pip3 install -q --no-cache-dir tqdm
pip3 install -q --no-cache-dir packaging # Required by mamba-ssm
pip3 install -q --no-cache-dir wheel # Required by mamba-ssm
pip3 install -q --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install -q --no-cache-dir --no-build-isolation mamba-ssm

# Submit your job to the cluster
## Parameters:
GRAPH_SIZE=100
PROBLEM_SIZE=100000
N_EPOCHS=10
BATCH_SIZE=256
BUFFER_SIZE=10000
TOUR_HEURISTIC="random"
ACTOR_LR=1e-4
ACTOR_LR_DECAY=0.99
CRITIC_LR=1e-4
CRITIC_LR_DECAY=0.99
REWARD_SCALE=1.0
LOSS_WEIGHT=0.5

DROPOUT=0.0
EMBEDDING_DIM=32
NUM_HARMONICS=32
MAMBA_HIDDEN_DIM=128
MAMBA_LAYERS=3
SCORE_HEAD_DIM=128
SCORE_HEAD_BIAS=True
NUM_ATTENTION_HEADS=8
FFN_EXPANSION=4
SINKHORN_TAU=0.5
SINKHORN_ITERS=10
TOUR_METHOD="greedy"
CONV_OUT_CHANNELS=128
CONV_KERNEL_SIZE=3
CONV_STRIDE=1
MLP_FF_DIM=512
MLP_EMBEDDING_DIM=128

srun --gres=gpu:1 python run.py --graph_size $GRAPH_SIZE --problem_size $PROBLEM_SIZE --n_epochs $N_EPOCHS --batch_size $BATCH_SIZE --buffer_size $BUFFER_SIZE --tour_heuristic $TOUR_HEURISTIC --actor_lr $ACTOR_LR --actor_lr_decay $ACTOR_LR_DECAY --critic_lr $CRITIC_LR --critic_lr_decay $CRITIC_LR_DECAY --reward_scale $REWARD_SCALE --loss_weight $LOSS_WEIGHT --dropout $DROPOUT --embedding_dim $EMBEDDING_DIM --num_harmonics $NUM_HARMONICS --mamba_hidden_dim $MAMBA_HIDDEN_DIM --mamba_layers $MAMBA_LAYERS --score_head_dim $SCORE_HEAD_DIM --score_head_bias $SCORE_HEAD_BIAS --num_attention_heads $NUM_ATTENTION_HEADS --ffn_expansion $FFN_EXPANSION --sinkhorn_tau $SINKHORN_TAU --sinkhorn_iters $SINKHORN_ITERS --tour_method $TOUR_METHOD --conv_out_channels $CONV_OUT_CHANNELS --conv_kernel_size $CONV_KERNEL_SIZE --conv_stride $CONV_STRIDE --mlp_ff_dim $MLP_FF_DIM --mlp_embedding_dim $MLP_EMBEDDING_DIM --no_progress_bar