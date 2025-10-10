import os
import time
import argparse
import torch


def get_options(args=None):
    
    parser = argparse.ArgumentParser(description="Route-Mamba")

    # Problem settings
    parser.add_argument('--problem', default='tsp', choices = ['vrp', 'tsp'], help="the targeted problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=100, help="the number of customers in the targeted problem (graph size)")
    parser.add_argument('--problem_size', type=int, default=10000, help='number of problem instances for training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')
    
    # Model parameters
    ## Common
    parser.add_argument('--tour_heuristic', type=str, default='greedy', help='Heuristic for initial tour construction (greedy, random, farthest)')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate for regularization (0 = no dropout)')
    ## Embedding Network
    parser.add_argument('--input_dim', type=int, default=2, help='input dimension of the problem nodes')
    parser.add_argument('--embedding_dim', type=int, default=32, help='dimension of embeddings for each, NFE & CE, has to be even')
    parser.add_argument('--num_harmonics', type=int, default=32, help='number of harmonics for cyclic positional encoding (recommended: <= N/2)')
    parser.add_argument('--frequency_scaling', type=float, default=0.0, help='How the amplitude should decay for harmonics with larger frequencies (between 0 and 1)')
    ## Mamba Encoder
    parser.add_argument('--mamba_hidden_dim', type=int, default=128, help='dimension of hidden state representation in Mamba')
    parser.add_argument('--mamba_layers', type=int, default=3, help='number of stacked Mamba blocks in the model')
    ## Bilinear Score Head (alternative to attention score head)
    parser.add_argument('--score_head_dim', type=int, default=128, help='dimension of the bilinear score head to construct score matrix')
    parser.add_argument('--score_head_bias', type=bool, default=True, help='whether to use bias in score head')
    ## Attention Score Head
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads in the model')
    parser.add_argument('--ffn_expansion', type=int, default=4, help='expansion factor for the FFN in the attention score head')
    parser.add_argument('--initial_identity_bias', type=float, default=10.0, help='initial bias to add to diagonal of score matrix to discourage self-loops')
    ## Gumbel-Sinkhorn Decoder
    parser.add_argument('--sinkhorn_tau', type=float, default=0.5, help='Sinkhorn temperature, higher = softer, lower = harder')
    parser.add_argument('--sinkhorn_tau_decay', type=float, default=0.9, help='Sinkhorn temperature decay rate per epoch')
    parser.add_argument('--sinkhorn_iters', type=int, default=10, help='Number of Sinkhorn iterations')
    ## Tour Constructor
    parser.add_argument('--tour_method', type=str, default='greedy', choices=['greedy', 'hungarian'], help='Method for tour construction')
    ## Convolutional Encoder (critic)
    parser.add_argument('--conv_out_channels', type=int, default=64, help='number of output channels for convolutional action encoder')
    parser.add_argument('--conv_kernel_size', type=int, default=3, help='kernel size for convolutional action encoder')
    parser.add_argument('--conv_stride', type=int, default=2, help='stride for convolutional action encoder')
    ## MLP Value Decoder (critic)
    parser.add_argument('--mlp_ff_dim', type=int, default=256, help='feed forward dimension for MLP value decoder')
    parser.add_argument('--mlp_embedding_dim', type=int, default=128, help='embedding dimension for MLP value decoder')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,help='number of instances per batch during training')
    parser.add_argument('--buffer_size', type=int, default=100000, help='size of the replay buffer')
    parser.add_argument('--actor_lr', type=float, default=1e-3, help="initial learning rate for the actor network")
    parser.add_argument('--actor_lr_decay', type=float, default=0.995, help='exponential learning rate decay per epoch')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help="initial learning rate for the critic network")
    parser.add_argument('--critic_lr_decay', type=float, default=0.995, help='exponential learning rate decay per epoch')
    parser.add_argument('--reward_scale', type=float, default=1.0, help='scaling factor for reward signal')
    parser.add_argument('--loss_weight', type=float, default=0.5, help='weighting factor between hard and soft critic loss (0 = only hard, 1 = only soft)')

    # Inference and validation parameters
    parser.add_argument('--eval_only', action='store_true', help='switch to inference mode')
    parser.add_argument('--eval_size', type=int, default=1000, help='number of instances for validation/inference')
    parser.add_argument('--val_dataset', type=str, default = './datasets/tsp_20_10000.pkl', help='dataset file path')

    # resume and load models
    parser.add_argument('--actor_load_path', default = None, help='path to load actor parameters from')
    parser.add_argument('--critic_load_path', default = None, help='path to load critic parameters from')
    parser.add_argument('--resume', default = None, help='resume from previous checkpoint file')
    parser.add_argument('--epoch_start', type=int, default=0, help='start at epoch # (relevant for learning rate decay)')

    # logs/output settings
    parser.add_argument('--no_progress_bar', action='store_true', help='disable progress bar')
    parser.add_argument('--log_dir', default='logs', help='directory to write TensorBoard information to')
    parser.add_argument('--log_step', type=int, default=100, help='log info every log_step gradient steps')
    parser.add_argument('--output_dir', default='outputs', help='directory to write output models to')
    parser.add_argument('--no_save', action='store_true', help='do not save models, only run inference')
    parser.add_argument('--run_name', default='run_name', help='name to identify the run')
    parser.add_argument('--checkpoint_epochs', type=int, default=0, help='save checkpoint every n epochs, 0 to save no checkpoints')
    
    opts = parser.parse_args(args)

    # processing settings
    opts.use_cuda = torch.cuda.is_available()
    opts.P = 250 if opts.eval_only else 1e10 # can set to smaller values e.g., 20 or 10, for generalization 
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not opts.resume else opts.resume.split('/')[-2]
    opts.save_dir = os.path.join(
        opts.output_dir,
        # "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    ) if not opts.no_save else None
    if opts.save_dir and not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir)

    return opts
