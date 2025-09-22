import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils import get_initial_tours, check_feasibility, compute_euclidean_tour


def save(model, optimizer, epoch, opts):
    """
    Saves the actor model's state_dict.
    Args:
        model: The actor model (nn.Module)
        optimizer: The optimizer (torch.optim.Optimizer)
        epoch: Current epoch number (int)
        opts: Model options (argparse.Namespace)
    """
    save_path = os.path.join(opts.save_dir, f"Model_{opts.problem}_epoch_{epoch+1}.pt")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")


def load(model, optimizer, opts, checkpoint_path):
    """
    Loads the actor model's state_dict and optimizer state from a checkpoint file.
    Args:
        model: The actor model (nn.Module)
        optimizer: The optimizer (torch.optim.Optimizer)
        opts: Model options (argparse.Namespace)
        checkpoint_path: Path to either the checkpoint file (.pt) or the directory containing it. If directory, latest checkpoint will be loaded.
    """
    if os.path.isdir(checkpoint_path):
        # If a directory is provided, load the latest checkpoint
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_path}")
        checkpoint_path = os.path.join(checkpoint_path, sorted(checkpoint_files)[-1])

    checkpoint = torch.load(checkpoint_path, map_location=opts.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Model loaded from {checkpoint_path}")


def validate(model, dataset, opts):
    """
    Validates the model on a given dataset.
    Args:
        model: The actor model (nn.Module)
        dataset: The validation dataset (torch.utils.data.Dataset)
        opts: Model options (argparse.Namespace)
    Returns:

    """
    dataloader = DataLoader(dataset, batch_size=opts.batch_size)
    model.eval()
    total_tour_length = 0
    total_initial_tour_length = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=opts.no_progress_bar):
            batch = {k: v.to(opts.device) for k, v in batch.items()}
            coords = batch['coordinates']

            # Create initial tour using specified heuristic (greedy or random)
            initial_tours = get_initial_tours(coords, opts.tour_heuristic)
            initial_tour_lengths = compute_euclidean_tour(initial_tours)

            # Forward pass to get the new tours
            new_tours = model(initial_tours)
            new_tour_lengths = compute_euclidean_tour(new_tours)

            total_tour_length += new_tour_lengths.sum().item()
            total_initial_tour_length += initial_tour_lengths.sum().item()
            total_samples += coords.size(0)

            # Feasibility check
            if not check_feasibility(new_tours):
                print("Warning: Infeasible tour detected during validation!")

    avg_initial_length = total_initial_tour_length / total_samples
    avg_new_length = total_tour_length / total_samples
    improvement = (avg_initial_length - avg_new_length) / avg_initial_length * 100

    print(f"Validation Results:", 
          f"- Avg Initial Tour Length: {avg_initial_length:.4f}, Avg New Tour Length: {avg_new_length:.4f}, Improvement: {improvement:.2f}%",
          f"- Total Samples: {total_samples}",
          sep="\n")


def train_epoch(model, optimizer, lr_scheduler, epoch, train_dataset, val_dataset, opts):
    """
    Performs one epoch of training.
    Args:
        model: The actor model (nn.Module)
        optimizer: The optimizer (torch.optim.Optimizer)
        lr_scheduler: Learning rate scheduler (torch.optim.lr_scheduler)
        epoch: Current epoch number (int)
        train_dataset: The training dataset (torch.utils.data.Dataset)
        val_dataset: The validation dataset (torch.utils.data.Dataset)
        opts: Model options (argparse.Namespace)
    """
    print("Start train epoch {} with lr={}".format(epoch, optimizer.param_groups[0]['lr']))
    step = epoch * (opts.problem_size // opts.batch_size)
    start_time = time.time()

    training_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
    )

    # Update Gumbel-Sinkhorn temperature
    model.decoder.gs_tau = max(opts.gs_tau_final, model.decoder.gs_tau / opts.anneal_rate)

    # Parameter Logging
    print("- Gumbel-Sinkhorn temperature: {:.4f}".format(model.decoder.gs_tau))
    print("- Learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))

    # Put model in train mode
    model.train()

    for _, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(model, optimizer, batch, step, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Validation
    validate(model, val_dataset, opts)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        save(model, optimizer, epoch, opts)


def train_batch(model, optimizer, batch, step, opts):

    batch = {k: v.to(opts.device) for k, v in batch.items()}
    coords = batch['coordinates']

    # Create initial tour using specified heuristic (greedy or random)
    initial_tours = get_initial_tours(coords, opts.tour_heuristic)
    initial_tour_lengths = compute_euclidean_tour(initial_tours)

    # Forward pass to get the new tours
    new_tours = model(initial_tours)
    new_tour_lengths = compute_euclidean_tour(new_tours)
    
    # Use the new tour length directly as the loss to minimize
    loss = new_tour_lengths.mean() * 100  # Apply loss scaling

    # Perform backward pass with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        parameters = model.parameters(), 
        max_norm = 1.0
    )

    # Logging
    if step % opts.log_step == 0:
        # Loss and tour lengths
        print(f"Step {step}: Loss = {loss.item():.4f}, Initial Tour Length = {initial_tour_lengths.mean().item():.4f}, New Tour Length = {new_tour_lengths.mean().item():.4f}")
        # Gradient existence check (for debugging)
        found_grads = 0
        vanishing_grads = 0
        total_params = len(list(model.parameters()))
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad:
                vanishing_grads += (param.grad.data.norm(2).item() <= 0.001)
                found_grads += 1
                grad_norms[name] = param.grad.data.norm(2).item()
        print(f"Step {step}: Found gradients for {found_grads}/{total_params} parameters, {vanishing_grads} with vanishing gradients.")

    # Step the optimizer
    optimizer.step()


