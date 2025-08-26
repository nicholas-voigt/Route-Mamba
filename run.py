import os
import json
import torch
import pprint
import numpy as np
import warnings
import random
from options import get_options

from problems.tsp import TSP
from agent.trainer import SurrogateLoss


def load_problem(name):
    problem = {
        'tsp': TSP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def load_agent(name):
    agent = {
        'surrogate': SurrogateLoss
    }.get(name, None)
    assert agent is not None, "Currently unsupported agent: {}!".format(name)
    return agent


def run(opts):

    # Pretty print the run args
    pprint.pprint(vars(opts))

    # Set the device - Has to be cuda, Mamba only works on GPU
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    if opts.device != torch.device("cuda"):
        raise RuntimeError("Mamba only works on GPU")

    # Figure out what's the problem
    problem = load_problem(opts.problem)(size=opts.graph_size)

    # Figure out the RL algorithm
    agent = load_agent(opts.RL_agent)(opts)

    # Load model checkpoint if specified
    if opts.load_path is not None:
        agent.load(opts.load_path)

    # Check if to evaluate only or train the model
    if opts.eval_only:
        # Start inference
        agent.evaluate(problem)        
    else:
        # Start training
        agent.train(problem)


if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run(get_options())
