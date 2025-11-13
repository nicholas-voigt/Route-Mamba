import torch
from torch import nn, optim
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


class ARPPOTrainer: 
    def __init__(self, opts):
        self.opts = opts

        # Initialize model with optimizer and learning rate scheduler
        if opts.actor_load_path:
            print(f"Loading actor model from {opts.actor_load_path}")
            self.model = torch.load(opts.actor_load_path, map_location=opts.device)
        else:
            self.model = Actor(
                input_dim = opts.input_dim,
                embed_dim = opts.embedding_dim,
                mamba_hidden_dim = opts.mamba_hidden_dim,
                mamba_layers = opts.mamba_layers,
                dropout = opts.dropout
            ).to(opts.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opts.actor_lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: opts.actor_lr_decay ** epoch)


    def train(self):
        torch.set_grad_enabled(True)
        torch.autograd.set_detect_anomaly(True)
        self.model.train()


    def eval(self):
        torch.set_grad_enabled(False)
        self.model.eval()


    def start_training(self, problem):
        self.gradient_check = False
        
        for epoch in range(self.opts.n_epochs):
            # prepare training dataset
            train_dataset = problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.problem_size)
            training_dataloader = DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size)

            # Logging
            print(f"\nTraining Epoch {epoch}:")
            print(f"-  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            logger = {
                'tour_cost': [],
                'baseline_cost': [],
                'actor_loss': [],
                'advantage': [],
                'ratios': [],
                'entropy': [],
                'clip_fraction': []
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
            print(f"-  Average Policy Ratios: {sum(logger['ratios'])/len(logger['ratios']):.4f}")
            print(f"-  Average Entropy: {sum(logger['entropy'])/len(logger['entropy']):.4f}")
            print(f"-  Clip Fraction: {sum(logger['clip_fraction'])/len(logger['clip_fraction']):.4f}")

            # update learning rates
            self.scheduler.step()

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
                torch.save(self.model, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
                print(f"Saved actor model at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, logger: dict):
        # --- 1. Collect Trajectories ---
        # get observations (Polar-ordered coordinates (canonical representation)) from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours) # (B, N, 2)

        # Actor forward pass
        with torch.no_grad():
            action, lp_sum, _ = self.model(observation)
            tours = torch.gather(observation, 1, action.unsqueeze(-1).expand(-1, -1, observation.size(-1)))
            # tours = torch.bmm(action.transpose(1, 2), observation)  # (B, N, 2)
            actor_cost = compute_euclidean_tour(tours)
            check_feasibility(observation, tours)

            # Store "old" policy
            old_lp_sum = lp_sum.detach()

        # --- 2. Calculate Advantage with GAE ---
        # Heuristic baseline (greedy)
        baseline_tours = get_heuristic_tours(observation, self.opts.baseline_tours)
        baseline_cost = compute_euclidean_tour(baseline_tours)

        # Normalized advantage over baseline
        raw_advantage = ((baseline_cost - actor_cost) / (baseline_cost + 1e-8))  # (B,)
        advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)  # Normalize advantage

        # --- 3. PPO Update (multiple epochs over collected data) ---
        ppo_metrics = {'actor_loss': [], 'entropy': [], 'ratios': [], 'clip_fraction': []}

        for _ in range(2):
            # Re-evaluate actions with current policy
            _, new_lp_sum, entropy = self.model(observation, actions=action)

            # Calculate ratio & clipped surrogate objective
            ratios = torch.exp(new_lp_sum - old_lp_sum)  # (B,)
            clip_param = 0.5
            actor_unclipped = -advantage * ratios
            actor_clipped = -advantage * torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
            actor_loss = torch.max(actor_unclipped, actor_clipped).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.gradient_check:
                log_gradients(self.model)
            self.gradient_check = False
            self.optimizer.step()

            # Logging
            with torch.no_grad():
                clip_fraction = ((ratios - 1.0).abs() > clip_param).float().mean()
            ppo_metrics['actor_loss'].append(actor_loss.item())
            ppo_metrics['entropy'].append(entropy.mean().item())
            ppo_metrics['ratios'].append(ratios.mean().item())
            ppo_metrics['clip_fraction'].append(clip_fraction.item())

        # Final logging after PPO epochs
        logger['tour_cost'].append(actor_cost.mean().item())
        logger['baseline_cost'].append(baseline_cost.mean().item())
        logger['advantage'].append(advantage.mean().item())
        logger['actor_loss'].append(sum(ppo_metrics['actor_loss']) / len(ppo_metrics['actor_loss']))
        logger['ratios'].append(sum(ppo_metrics['ratios']) / len(ppo_metrics['ratios']))
        logger['entropy'].append(sum(ppo_metrics['entropy']) / len(ppo_metrics['entropy']))
        logger['clip_fraction'].append(sum(ppo_metrics['clip_fraction']) / len(ppo_metrics['clip_fraction']))

