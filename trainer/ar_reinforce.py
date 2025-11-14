import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

import components as mc
from utils.utils import compute_euclidean_tour, get_heuristic_tours, check_feasibility
from utils.logger import log_gradients


class Actor(nn.Module):
    def __init__(self, input_dim, embed_dim, mamba_hidden_dim, mamba_layers, dropout):
        super().__init__()

        # Model components
        self.feature_embedder = mc.StructuralEmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embed_dim
        )
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.encoder = mc.BidirectionalMambaEncoder(
            mamba_model_size = embed_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.encoder_norm = nn.LayerNorm(2 * embed_dim)
        self.policy_decoder = mc.ARPointerDecoder(embedding_dim = 2 * embed_dim)

    def forward(self, batch: torch.Tensor, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
            actions: (B, N, N) - previously taken actions as permutation (for training)
        Returns:
            tours: (B, N) - node indices representing the tour
            log_probs: (B,) - log probability of the entire tour
            entropies: (B,) - entropy of the tours probability distribution
        """
        # 1. Create Embeddings & normalize
        embeddings = self.feature_embedder(batch)  # (B, N, E)
        embeddings = self.embedding_norm(embeddings)  # (B, N, E)

        # 2. Encoder: Layered Mamba blocks with internal Pre-LN
        encoded_features = self.encoder(embeddings)
        encoded_features = self.encoder_norm(encoded_features)   # (B, N, 2E)
        graph_embed = encoded_features.mean(dim=1)  # (B, 2E)

        # 3. Score Construction: Decode tours using Mamba + attention + Sinkhorn
        tour_idxs, log_probs, entropies = self.policy_decoder(graph_embed, encoded_features, actions)

        return tour_idxs, torch.sum(log_probs, dim=1), torch.sum(entropies, dim=1)


class ARTrainer:
    def __init__(self, opts) -> None:
        self.opts = opts

        # Initialize the actor with optimizer and learning rate scheduler
        self.actor = Actor(
            input_dim = opts.input_dim,
            embed_dim = opts.embedding_dim,
            mamba_hidden_dim = opts.mamba_hidden_dim,
            mamba_layers = opts.mamba_layers,
            dropout = opts.dropout
        ).to(opts.device)

        if opts.actor_load_path:
            print(f"Loading actor model from {opts.actor_load_path}")
            self.actor = torch.load(opts.actor_load_path, map_location=opts.device, weights_only=False)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=opts.actor_lr)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lambda epoch: opts.actor_lr_decay ** epoch)


    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()


    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()


    def start_training(self, problem):
        self.gradient_check = False
        
        for epoch in range(self.opts.n_epochs):
            # prepare training dataset
            train_dataset = problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.problem_size)
            training_dataloader = DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size)

            # Logging
            print(f"\nTraining Epoch {epoch}:")
            print(f"-  Actor Learning Rate: {self.actor_optimizer.param_groups[0]['lr']:.6f}")

            logger = {
                'tour_cost': [],
                'baseline_cost': [],
                'actor_loss': [],
                'advantage': [],
                'log_likelihood': [],
                'entropy': []
            }

            # training batch loop
            self.train()
            self.gradient_check = True
            start_time = time.time()

            for _, batch in enumerate(tqdm(training_dataloader, disable=self.opts.no_progress_bar)):
                self.train_batch(batch, logger)

            epoch_duration = time.time() - start_time
            print(f"Training Epoch {epoch} completed. Results:")
            print(f"-  Epoch Runtime: {epoch_duration:.2f}s")
            print(f"-  Average Tour Cost: {sum(logger['tour_cost'])/len(logger['tour_cost']):.4f}")
            print(f"-  Average Baseline Cost: {sum(logger['baseline_cost'])/len(logger['baseline_cost']):.4f}")
            print(f"-  Average Actor Loss: {sum(logger['actor_loss'])/len(logger['actor_loss']):.4f}")
            print(f"-  Average Advantage: {sum(logger['advantage'])/len(logger['advantage']):.4f}")
            print(f"-  Average Log Likelihood: {sum(logger['log_likelihood'])/len(logger['log_likelihood']):.4f}")
            print(f"-  Average Entropy: {sum(logger['entropy'])/len(logger['entropy']):.4f}")

            # update learning rates
            self.actor_scheduler.step()

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
                torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
                print(f"Saved actor model at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, logger: dict, warmup_mode: bool = False):
        # get observations (Polar-ordered coordinates (canonical representation)) from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours) # (B, N, 2)

        # Actor forward pass
        tour_indices, lp_sums, entropy_sums = self.actor(observation) # (B, N), (B,), (B,)
        tours = torch.gather(observation, 1, tour_indices.unsqueeze(-1).expand(-1, -1, observation.size(-1)))  # (B, N, 2)
        tour_cost = compute_euclidean_tour(tours)

        check_feasibility(observation, tours)

        # Heuristic baseline (greedy)
        baseline_tours = get_heuristic_tours(observation, self.opts.baseline_tours)
        baseline_cost = compute_euclidean_tour(baseline_tours)

        # Actor loss using REINFORCE with heuristic baseline
        with torch.no_grad():
            advantage = (tour_cost - baseline_cost)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = (advantage * lp_sums).mean()

        # Logging
        logger['tour_cost'].append(tour_cost.mean().item())
        logger['baseline_cost'].append(baseline_cost.mean().item())
        logger['actor_loss'].append(actor_loss.item())
        logger['advantage'].append(advantage.mean().item())
        logger['log_likelihood'].append(lp_sums.mean().item())
        logger['entropy'].append(entropy_sums.mean().item())

        # Actor Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if self.gradient_check:
            log_gradients(self.actor)
        self.gradient_check = False
        self.actor_optimizer.step()


    def start_evaluation(self, problem):
        # prepare dataset
        val_dataset = problem.make_dataset(
            size=self.opts.graph_size, num_samples=self.opts.eval_size, filename=self.opts.val_dataset)
        dataloader = DataLoader(val_dataset, batch_size=self.opts.batch_size)

        # Logging
        print("\nStarting Evaluation:")
        print(f"-  Evaluating {self.opts.problem}-{self.opts.graph_size}")
        print(f"-  Eval Dataset Size: {len(val_dataset)}")

        logger = {
            'tour_cost': [],
            'baseline_cost': [],
        }

        # start evaluation loop
        self.eval()
        start_time = time.time()

        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, disable=self.opts.no_progress_bar)):
                self.evaluate_batch(batch, logger)

        end_time = time.time() - start_time
        print(f"Evaluation completed. Results:")
        print(f"-  Runtime: {end_time:.2f}s")
        print(f"-  Average Tour Cost: {sum(logger['tour_cost'])/len(logger['tour_cost']):.4f}")
        print(f"-  Average Baseline Cost: {sum(logger['baseline_cost'])/len(logger['baseline_cost']):.4f}")
    

    def evaluate_batch(self, batch, logger):
        # get observations (Polar-ordered coordinates (canonical representation)) from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours) # (B, N, 2)

        # Actor forward pass
        tour_indices, _, _ = self.actor(observation) # (B, N), (B,), (B,)
        tours = torch.gather(observation, 1, tour_indices.unsqueeze(-1).expand(-1, -1, observation.size(-1)))  # (B, N, 2)
        tour_cost = compute_euclidean_tour(tours)

        check_feasibility(observation, tours)

        # Heuristic baseline (greedy)
        baseline_tours = get_heuristic_tours(observation, self.opts.baseline_tours)
        baseline_cost = compute_euclidean_tour(baseline_tours)

        # Logging
        logger['tour_cost'].append(tour_cost.mean().item())
        logger['baseline_cost'].append(baseline_cost.mean().item())
