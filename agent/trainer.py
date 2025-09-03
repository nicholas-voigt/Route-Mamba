import os
import torch
from torch.utils.data import DataLoader

from nets.actor_network import Actor
from problems.tsp import TSP, TSPDataset
from utils import get_initial_tours, check_feasibility, compute_euclidean_tour


class SurrogateLoss:
    def __init__(self, opts):
        self.opts = opts
        # Initialize actor network
        self.actor = Actor(
            input_dim = opts.problem_input_dim,
            embedding_dim = opts.embedding_dim,
            harmonics = opts.harmonics,
            frequency_scaling = opts.frequency_scaling,
            model_dim = opts.model_dim,
            hidden_dim = opts.hidden_dim,
            mamba_layers = opts.mamba_layers,
            score_head_dim = opts.score_head_dim,
            score_head_bias = opts.score_head_bias,
            gs_tau = opts.gs_tau_initial,
            gs_iters = opts.gs_iters,
            method = opts.tour_method,
            device = opts.device
        ).to(opts.device)
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            params = self.actor.parameters(),
            lr = opts.lr_model
        )
        # Initialize simple logger
        self.training_log = []

    def save(self, epoch, save_dir, path_prefix="checkpoint"):
        """
        Saves the actor model's state_dict after each epoch.
        Args:
            epoch: Current epoch number (int)
            path_prefix: Prefix for the checkpoint file (default: "checkpoint")
        """
        save_path = os.path.join(save_dir, f"{path_prefix}_{self.opts.run_name}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'opts': self.opts
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, checkpoint_path):
        """
        Loads the actor model's state_dict and optimizer state from a checkpoint file.
        Args:
            checkpoint_path: Path to either the checkpoint file (.pt) or the directory containing it. If directory, latest checkpoint will be loaded.
        """
        if os.path.isdir(checkpoint_path):
            # If a directory is provided, load the latest checkpoint
            checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_path}")
            checkpoint_path = os.path.join(checkpoint_path, sorted(checkpoint_files)[-1])

        checkpoint = torch.load(checkpoint_path, map_location=self.opts.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {checkpoint_path}")

    def train(self, problem: TSP):
        """
        Minimal MVP training loop with greedy initial tour.
        Args:
            problem: The TSP problem instance.
        """
        torch.manual_seed(self.opts.seed)

        # Initialize Gumbel-Sinkhorn parameters
        anneal_rate = (self.opts.gs_tau_initial / self.opts.gs_tau_final) ** (1.0 / self.opts.n_epochs)

        for epoch in range(self.opts.n_epochs):
            # prepare training data
            training_dataset = problem.make_dataset(
                size=self.opts.graph_size,
                num_samples=self.opts.epoch_size
            )
            training_dataloader = DataLoader(
                dataset=training_dataset,
                batch_size=self.opts.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            # Initialize accumulators for logging
            total_loss = 0
            total_initial_length = 0
            total_new_length = 0
            num_samples = len(training_dataloader.dataset)  # type: ignore
            # Update Gumbel-Sinkhorn temperature
            self.actor.decoder.gs_tau = max(self.opts.gs_tau_final, self.actor.decoder.gs_tau / anneal_rate)

            # Batch training loop
            for batch in training_dataloader:
                self.actor.train()
                batch = {k: v.to(self.opts.device) for k, v in batch.items()}
                coords = batch['coordinates']

                # Create initial tour using specified heuristic (greedy or random)
                initial_tours = get_initial_tours(coords, self.opts.tour_heuristic)

                # Forward pass to get the straight-through permutation
                st_perm = self.actor(initial_tours)

                # Create the new tour using the permutation & compute tour lengths
                new_tours = torch.bmm(st_perm, initial_tours)
                initial_tour_lengths = compute_euclidean_tour(initial_tours)
                new_tour_lengths = compute_euclidean_tour(new_tours)
                
                # Use the new tour length directly as the loss to minimize
                loss = new_tour_lengths.mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics for logging
                total_loss += loss.item() * coords.size(0)
                total_initial_length += initial_tour_lengths.sum().item()
                total_new_length += new_tour_lengths.sum().item()

            # 3. Calculate epoch averages and log them
            avg_loss = total_loss / num_samples
            avg_initial_length = total_initial_length / num_samples
            avg_new_length = total_new_length / num_samples

            epoch_log = {
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
                'avg_initial_length': avg_initial_length,
                'avg_new_length': avg_new_length
            }
            self.training_log.append(epoch_log)

            print(
                f"Epoch {epoch+1}/{self.opts.n_epochs} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Initial Length: {avg_initial_length:.4f} | "
                f"New Length: {avg_new_length:.4f}"
            )
            
            # Save model checkpoint
            if self.opts.checkpoint_epochs and (epoch + 1) % self.opts.checkpoint_epochs == 0:
                self.save(epoch=epoch, save_dir=self.opts.save_dir)

    def evaluate(self, problem: TSP):
        """
        MVP evaluation loop: runs inference (greedy initial tour construction plus soft actor-critic), checks feasibility, and reports average tour length.
        Uses the Hungarian algorithm for hard permutation extraction.
        Args:
            problem: The TSP problem instance.
        """
        self.actor.eval()
        torch.manual_seed(self.opts.seed)

        # Prepare evaluation data
        eval_dataset = problem.make_dataset(
            size=self.opts.graph_size,
            num_samples=self.opts.eval_size
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.opts.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        total_length = 0
        total_feasible = 0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.opts.device) for k, v in batch.items()}
                coords = batch['coordinates']

                # 1. Create initial tour using specified heuristic (greedy or random)
                initial_tours = get_initial_tours(coords, self.opts.tour_heuristic)

                # 2. Forward pass to get the straight-through permutation & check feasibility (sanity check)
                st_perm = self.actor(initial_tours)
                check_feasibility(st_perm)

                # 3. Create the new tour using the permutation & calculate tour lengths
                new_tours = torch.bmm(st_perm, initial_tours)   # (B, N, 2)
                new_tour_lengths = compute_euclidean_tour(new_tours)  # (B,)

                # 4. Accumulate statistics
                total_length += new_tour_lengths.sum().item()
                total_feasible += new_tour_lengths.size(0)
                total_samples += new_tour_lengths.size(0)

        avg_length = total_length / total_samples if total_samples > 0 else float('inf')
        print(f"Evaluation: Avg Tour Length: {avg_length:.4f} | Feasible Tours: {total_feasible}/{total_samples}")