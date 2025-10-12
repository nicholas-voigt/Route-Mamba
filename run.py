import os
import pprint
import warnings
import torch
from torch import optim

from problems.tsp import TSP
from trainer.spg import SPGTrainer
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

    # Initialize the Agent, which includes actor, critic, optimizers, schedulers and memory buffer
    agent = SPGTrainer(opts)

    # Start the actual training/inference loop
    if opts.eval_only:
        agent.start_evaluation(problem)
    else:
        agent.start_training(problem)

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(get_options())
