import os
import pprint
import warnings
import torch
from torch import optim

from problems.tsp import TSP
from nets.actor_network import Actor
from trainer import train_epoch, validate
from options import get_options

def load_problem(name):
    problem = {
        'tsp': TSP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Set the device - Has to be cuda, Mamba only works on GPU
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    if opts.device != torch.device("cuda"):
        raise RuntimeError("Mamba only works on GPU")

    # Figure out what's the problem
    problem = load_problem(opts.problem)(size=opts.graph_size)

    # Initialize the model
    model = Actor(
        input_dim = opts.input_dim,
        embedding_dim = opts.embedding_dim,
        num_harmonics = opts.num_harmonics,
        frequency_scaling = opts.frequency_scaling,
        mamba_hidden_dim = opts.mamba_hidden_dim,
        mamba_layers = opts.mamba_layers,
        num_attention_heads = opts.num_attention_heads,
        gs_tau = opts.gs_tau_initial,
        gs_iters = opts.gs_iters,
        method = opts.tour_method,
    ).to(opts.device)

    # Initialize optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=opts.initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training/inference loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset)
    train_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.problem_size)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.n_epochs):
            train_epoch(model, optimizer, scheduler, epoch, train_dataset, val_dataset, opts)


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(get_options())
