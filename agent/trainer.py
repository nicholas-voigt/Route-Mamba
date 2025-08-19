import os
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from nets.actor_network import Actor
from problems.tsp import TSP, TSPDataset
from utils import greedy_initial_tour, check_feasibility, compute_soft_tour_length


class SurrogateLoss:
    def __init__(self, size, opts):
        self.opts = opts
        self.actor = Actor(
            input_dim = opts.problem_input_dim,
            embedding_dim = opts.embedding_dim,
            frequency_base = opts.frequency_base,
            freq_spread = opts.freq_spread,
            model_dim = opts.model_dim,
            hidden_dim = opts.hidden_dim,
            score_dim = opts.score_dim,
            gs_tau = opts.gs_tau,
            gs_iters = opts.gs_iters,
            seq_length = size,
            device = opts.device
        ).to(opts.device)
        self.optimizer = torch.optim.Adam(
            params = self.actor.parameters(),
            lr = opts.lr_model
        )

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

            # Batch training loop
            total_loss = 0
            for batch in training_dataloader:
                self.actor.train()
                batch = {k: v.to(self.opts.device) for k, v in batch.items()}
                coords = batch['coordinates']

                # 1. Create initial tour using greedy heuristic
                batch_init = greedy_initial_tour(coords)  # (batch_size, seq_length, 2)

                # 2. Forward pass: get soft tour and permutation
                soft_tour, _ = self.actor(batch_init)

                # 3. Compute surrogate loss (tour length)
                tour_length = compute_soft_tour_length(soft_tour)
                loss = tour_length.mean()

                # 5. Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch['coordinates'].size(0)

            avg_loss = total_loss / len(training_dataloader.dataset) # type: ignore
            print(f"Epoch {epoch+1}/{self.opts.n_epochs} | Avg Loss: {avg_loss:.4f}")
            
            # Save model checkpoint
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

                # 1. Create initial tour using greedy heuristic
                batch_init = greedy_initial_tour(coords)  # (batch_size, seq_length, 2)

                # 2. Forward pass: get soft tour and permutation
                _, soft_perm = self.actor(batch_init)

                # 3. Use Hungarian algorithm to extract hard permutation indices
                batch_size, _, _ = soft_perm.size()
                perm_indices = []
                for i in range(batch_size):
                    # Hungarian algorithm expects cost matrix, so use -soft_perm
                    _, col_ind = linear_sum_assignment(-soft_perm[i].cpu().numpy())
                    # col_ind gives the assignment for each row
                    perm_indices.append(torch.tensor(col_ind, dtype=torch.long, device=coords.device))
                perm_indices = torch.stack(perm_indices, dim=0)  # (batch_size, seq_length)

                # 4. Check feasibility
                check_feasibility(perm_indices)

                # 5. Reorder coordinates according to true permutation & compute tour length
                ordered_coords = coords.gather(1, perm_indices.unsqueeze(-1).expand(-1, -1, 2))
                tour_length = compute_soft_tour_length(ordered_coords)  # (batch_size,)

                # 6. Accumulate statistics
                total_length += tour_length.sum().item()
                total_feasible += perm_indices.size(0)
                total_samples += perm_indices.size(0)

        avg_length = total_length / total_samples if total_samples > 0 else float('inf')
        print(f"Evaluation: Avg Tour Length: {avg_length:.4f} | Feasible Tours: {total_feasible}/{total_samples}")