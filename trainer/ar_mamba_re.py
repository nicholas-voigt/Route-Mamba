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
    def __init__(self, input_dim, embed_dim, mamba_hidden_dim, mamba_layers, gs_tau, gs_iters, tour_method, dropout):
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
        self.policy_decoder = mc.ARMambaDecoder(
            embed_dim = 2 * embed_dim,
            mamba_hidden_dim = mamba_hidden_dim,
            mamba_layers = mamba_layers,
            key_proj_bias = False,
            dropout = dropout
        )
        self.score_norm = mc.GumbelSinkhornDecoder(
            gs_tau = gs_tau,
            gs_iters = gs_iters
        )
        self.tour_constructor = mc.TourConstructor(tour_method)

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
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
        logits = self.policy_decoder(graph_embed, encoded_features)

        # 4. Tour Construction: Sample or reconstruct tours
        tour_perms = self.tour_constructor(logits)
        log_probs = torch.sum(logits * tour_perms, dim=(1, 2))  # (B,)
        probs = torch.exp(logits)
        entropies = -torch.sum(logits * probs, dim=(1, 2))  # (B,)

        return tour_perms, log_probs, entropies


class ARMambaTrainer:
    def __init__(self, opts) -> None:
        self.opts = opts

        # Initialize the actor with optimizer and learning rate scheduler
        if opts.actor_load_path:
            print(f"Loading actor model from {opts.actor_load_path}")
            self.actor = torch.load(opts.actor_load_path, map_location=opts.device)
        else:
            self.actor = Actor(
                input_dim = opts.input_dim,
                embed_dim = opts.embedding_dim,
                mamba_hidden_dim = opts.mamba_hidden_dim,
                mamba_layers = opts.mamba_layers,
                gs_tau = opts.sinkhorn_tau,
                gs_iters = opts.sinkhorn_iters,
                tour_method = opts.tour_method,
                dropout = opts.dropout
            ).to(opts.device)
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
        actions, lp_sums, entropy_sums = self.actor(observation) # (B, N), (B,), (B,)
        tours = torch.bmm(actions.transpose(1, 2), observation)  # (B, N, 2)
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
        self.eval()

        # Logging
        logger = {
            'initial_tour_length': [],
            'new_tour_length': [],
            'reward': []
        }

        # start evaluation loop
        print("\nStarting Evaluation:")
        print(f"-  Evaluating {self.opts.problem}-{self.opts.graph_size}")
        print(f"-  Eval Dataset Size: {len(val_dataset)}")

        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, disable=self.opts.no_progress_bar)):
                self.evaluate_batch(batch, logger)
    

    def evaluate_batch(self, batch, logger):
        # get observations (initial tours) through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        coords = batch['coordinates']
        initial_tours = get_heuristic_tours(coords, self.opts.tour_heuristic)
        initial_tour_lengths = compute_euclidean_tour(initial_tours)

        # Actor forward pass & tour construction & reward calculation
        dense_actions, discrete_actions = self.actor(initial_tours)

        # Reward calculation, TODO: Include expected reward for soft actions?
        new_tour_lengths = compute_euclidean_tour(torch.bmm(discrete_actions.transpose(1, 2), initial_tours))
        reward = (initial_tour_lengths - new_tour_lengths) * self.opts.reward_scale  # Apply reward scaling
