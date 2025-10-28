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
#SBATCH --constraint="EYPC&48gb"

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
pip3 install -q --no-cache-dir numpy
pip3 install -q --no-cache-dir scipy
pip3 install -q --no-cache-dir tqdm
pip3 install -q --no-cache-dir packaging # Required by mamba-ssm
pip3 install -q --no-cache-dir wheel # Required by mamba-ssm
pip3 install -q --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install -q --no-cache-dir --no-build-isolation mamba-ssm

# Submit your job to the cluster
## Parameters:
TRAINER="spg"
GRAPH_SIZE=20
PROBLEM_SIZE=200000
N_EPOCHS=10
BATCH_SIZE=512
BUFFER_SIZE=10000
INITIAL_TOURS="polar"
BASELINE_TOURS="greedy"
ACTOR_LR=5e-3
ACTOR_LR_DECAY=0.98
CRITIC_LR=1e-4
CRITIC_LR_DECAY=0.95
REWARD_SCALE=1.0
EPSILON=0.3
LAMBDA_AUXILIARY_LOSS=0.1

DROPOUT=0.1
EMBEDDING_DIM=64
KNN_NEIGHBORS=8
MAMBA_HIDDEN_DIM=256
MAMBA_LAYERS=3
SCORE_HEAD_DIM=128
SCORE_HEAD_BIAS=False
NUM_ATTENTION_HEADS=8
FFN_EXPANSION=4
INITIAL_IDENTITY_BIAS=2.0
SINKHORN_TAU=3.0
SINKHORN_TAU_DECAY=0.95
SINKHORN_ITERS=10
TOUR_METHOD="greedy"
MLP_FF_DIM=32
MLP_EMBEDDING_DIM=16

srun --gres=gpu:1 python run.py --trainer $TRAINER --graph_size $GRAPH_SIZE --problem_size $PROBLEM_SIZE --n_epochs $N_EPOCHS --batch_size $BATCH_SIZE --buffer_size $BUFFER_SIZE --initial_tours $INITIAL_TOURS --baseline_tours $BASELINE_TOURS --actor_lr $ACTOR_LR --actor_lr_decay $ACTOR_LR_DECAY --critic_lr $CRITIC_LR --critic_lr_decay $CRITIC_LR_DECAY --reward_scale $REWARD_SCALE --epsilon $EPSILON --lambda_auxiliary_loss $LAMBDA_AUXILIARY_LOSS --dropout $DROPOUT --embedding_dim $EMBEDDING_DIM --kNN_neighbors $KNN_NEIGHBORS --mamba_hidden_dim $MAMBA_HIDDEN_DIM --mamba_layers $MAMBA_LAYERS --score_head_dim $SCORE_HEAD_DIM --score_head_bias $SCORE_HEAD_BIAS --num_attention_heads $NUM_ATTENTION_HEADS --ffn_expansion $FFN_EXPANSION --initial_identity_bias $INITIAL_IDENTITY_BIAS --sinkhorn_tau $SINKHORN_TAU --sinkhorn_tau_decay $SINKHORN_TAU_DECAY --sinkhorn_iters $SINKHORN_ITERS --tour_method $TOUR_METHOD --mlp_ff_dim $MLP_FF_DIM --mlp_embedding_dim $MLP_EMBEDDING_DIM --no_progress_bar