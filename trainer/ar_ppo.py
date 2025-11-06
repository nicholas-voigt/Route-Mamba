import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import time
from tqdm import tqdm

import components as mc
from utils.utils import compute_euclidean_tour, get_heuristic_tours
from utils.logger import log_gradients


class PolicyDecoder(nn.Module):
    def __init__(self, embed_dim: int, context_dim: int, mamba_hidden_dim: int, key_proj_bias: bool, dropout: float, gs_tau: float, gs_iters: int):
        """
        Decoder for TSP in autoregressive style but faster with mamba scan.
        Takes node embeddings and graph embeddings as input and produces a tour.
        Uses Mamba for context encoding and query formulation and Transformer-style attention for pointing.
        Args:
            embedding_dim: Dimension of the input embeddings for one node
            mamba_hidden_dim: Hidden state size for Mamba
            key_proj_bias: Bias for key projection layer
            dropout: Dropout rate for regularization
        """
        super(PolicyDecoder, self).__init__()
        self.key_projection = nn.Linear(embed_dim, context_dim, bias=key_proj_bias)
        self.query_projection = mc.MambaBlock(
            mamba_model_size = context_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout
        )
        self.sinkhorn_decoder = mc.GumbelSinkhornDecoder(
            gs_tau = gs_tau,
            gs_iters = gs_iters
        )

    def forward(self, graph_emb: torch.Tensor, node_emb: torch.Tensor, actions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the autoregressive pointer decoder.
        Args:
            graph_emb: (B, 2E) - graph embedding
            node_emb: (B, N, E) - node embeddings (for key projection)
            actions: (B, N) - previously taken actions (for training)
        Returns:
            tour: (B, N) - tensor of node indices representing the tour
            log_probs: (B, N) - log-probability of each selected node at each step
            entropies: (B, N) - entropy of the probability distribution at each step
        """
        B, N, _ = node_emb.shape
        device = node_emb.device
        NEG = torch.finfo(node_emb.dtype).min
        batch_indices = torch.arange(B, device=device)

        # Tour construction:
        # Project node embeddings to keys & concatenated graph and node embeddings to queries
        keys = self.key_projection(node_emb) # (B, N, C)
        queries = self.query_projection(torch.cat([graph_emb.unsqueeze(1).expand(-1, N, -1), node_emb], dim=-1))  # (B, N, C)

        # Calculate attention scores (logits) for all nodes in one go & 
        # normalize matrix using Sinkhorn operator to get doubly stochastic matrix
        logits = torch.bmm(keys, queries.transpose(1, 2))  # (B, N, C) @ (B, C, N) -> (B, N, N)
        norm_logits = self.sinkhorn_decoder(logits)
        
        # Construct tour by sampling from the normalized logits
        tour_perms = mc.TourConstructor(method='sampled')(norm_logits)
        tours = tour_perms.argmax(dim=-1)  # (B, N)
        log_probs = torch.gather(norm_logits, 2, tour_perms).squeeze(-1)  # (B, N)
        entropies = -torch.sum(norm_logits * torch.exp(norm_logits), dim=-1)  # (B, N)

        return tours, log_probs, entropies


class ActorCritic(nn.Module):
    def __init__(self, input_dim, embed_dim, mamba_hidden_dim, mlp_ff_dim, mlp_embed_dim, dropout):
        super().__init__()

        # Model components
        self.feature_embedder = mc.StructuralEmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embed_dim
        )
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.policy_head = PolicyDecoder(
            embed_dim = embed_dim,
            context_dim = 2 * embed_dim,
            mamba_hidden_dim = mamba_hidden_dim,
            key_proj_bias = False,
            dropout = dropout,
            gs_tau = 2.0,
            gs_iters = 10
        )
        self.critic_head = mc.MLP(
            input_dim = embed_dim,
            feed_forward_dim = mlp_ff_dim,
            embedding_dim = mlp_embed_dim,
            output_dim = 1,
            dropout = dropout
        )
    
    def get_embeddings(self, batch: torch.Tensor):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
        Returns:
            node_embed: (B, N, E) - node embeddings
            graph_embed: (B, E) - graph embeddings
        """
        node_embed = self.feature_embedder(batch)  # (B, N, E)
        node_embed = self.embedding_norm(node_embed)  # (B, N, E)
        graph_embed = node_embed.mean(dim=1)  # (B, E)
        return node_embed, graph_embed

    def get_values(self, batch: torch.Tensor):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
        Returns:
            state_values: (B,) - estimated state values for the input graphs
        """
        _, graph_embed = self.get_embeddings(batch)
        return self.critic_head(graph_embed).squeeze(1)  # (B,)
    
    def get_actions(self, batch: torch.Tensor, actions: torch.Tensor | None = None):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
            actions: (B, N) - previously taken actions (for training)
        Returns:
            tours: (B, N) - node indices representing the tour
            logits: (B, N) - log probability of chosen nodes at each step
            entropies: (B, N) - entropy of the probability distribution at each step
        """
        node_embed, graph_embed = self.get_embeddings(batch)
        tours, logits, entropies = self.policy_head(graph_embed, node_embed, actions)  # (B, N), (B, N), (B, N)
        return tours, logits, entropies

    def forward(self, batch: torch.Tensor, actions: torch.Tensor | None = None):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
            actions: (B, N) - previously taken actions (for training)
        Returns:
            tours: (B, N) - node indices representing the tour
            logits: (B, N) - log probability of chosen nodes at each step
            entropies: (B, N) - entropy of the probability distribution at each step
            state_values: (B,) - estimated state values for the input graphs
        """
        # 1. Encoder: Encode Input & normalize
        node_embed, graph_embed = self.get_embeddings(batch)
        # 2. Policy Decoder: Autoregressive Pointer Network
        tours, logits, entropies = self.policy_head(graph_embed, node_embed, actions)  # (B, N), (B, N), (B, N)
        # 3. Critic Head: Estimate state value
        state_values = self.critic_head(graph_embed).squeeze(1)  # (B,)
        return tours, logits, entropies, state_values


class ARPPOTrainer: 
    def __init__(self, opts):
        self.opts = opts

        # Initialize model with optimizer and learning rate scheduler
        if opts.actor_load_path:
            print(f"Loading actor model from {opts.actor_load_path}")
            self.model = torch.load(opts.actor_load_path, map_location=opts.device)
        else:
            self.model = ActorCritic(
                input_dim = opts.input_dim,
                embed_dim = opts.embedding_dim,
                mamba_hidden_dim = opts.mamba_hidden_dim,
                mlp_ff_dim = opts.mlp_ff_dim,
                mlp_embed_dim = opts.mlp_embedding_dim,
                dropout = opts.dropout
            ).to(opts.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opts.actor_lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: opts.actor_lr_decay ** epoch)


    def train(self):
        torch.set_grad_enabled(True)
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
                'critic_cost': [],
                'actor_loss': [],
                'critic_loss': [],
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
            print(f"-  Average Critic Cost: {sum(logger['critic_cost'])/len(logger['critic_cost']):.4f}")
            print(f"-  Average Actor Loss: {sum(logger['actor_loss'])/len(logger['actor_loss']):.4f}")
            # print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")
            print(f"-  Average Advantage: {sum(logger['advantage'])/len(logger['advantage']):.4f}")
            print(f"-  Average Policy Ratios: {sum(logger['ratios'])/len(logger['ratios']):.4f}")
            print(f"-  Average Entropy: {sum(logger['entropy'])/len(logger['entropy']):.4f}")
            print(f"-  Clip Fraction: {sum(logger['clip_fraction'])/len(logger['clip_fraction']):.4f}")

            # update learning rates
            self.scheduler.step()

            # if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
            #     torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
            #     torch.save(self.critic, f"{self.opts.save_dir}/critic_{self.opts.problem}_epoch{epoch + 1}.pt")
            #     print(f"Saved actor and critic models at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, logger: dict):
        # --- 1. Collect Trajectories ---
        # get observations (initial tours) through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours) # (B, N, 2)
        B, N, _ = observation.size()

        # Actor forward pass
        with torch.no_grad():
            action, log_probs, _ = self.model.get_actions(observation)
            tours = torch.gather(observation, 1, action.unsqueeze(-1).expand(-1, -1, observation.size(-1)))  # (B, N, 2)
            actor_cost = compute_euclidean_tour(tours)

            # Heuristic baseline instead of learned critic
            baseline_tours = get_heuristic_tours(observation, self.opts.baseline_tours)
            values = compute_euclidean_tour(baseline_tours)

            # Store "old" policy
            old_lp_sum = torch.sum(log_probs, dim=1).detach()

        # --- 2. Calculate Advantage with GAE ---
        # Relative advantage over critic estimation
        advantage = values - actor_cost  # (B,)
        advantage = ((advantage + actor_cost) / actor_cost) * self.opts.reward_scale  # Apply reward scaling

        logger['tour_cost'].append(actor_cost.mean().item())
        logger['critic_cost'].append(values.mean().item())

        # --- 3. PPO Update (multiple epochs over collected data) ---
        for _ in range(4):
            # Re-evaluate actions with current policy
            _, new_log_probs, entropy = self.model.get_actions(observation, actions=action)
            new_lp_sum = torch.sum(new_log_probs, dim=1)  # (B,)

            # Calculate ratio & clipped surrogate objective
            ratios = torch.exp(new_lp_sum - old_lp_sum)  # (B,)
            clip_param = 0.2
            actor_unclipped = -advantage * ratios
            actor_clipped = -advantage * torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
            actor_loss = torch.max(actor_unclipped, actor_clipped).mean()

            # Critic loss (MSE)
            # critic_unclipped = F.mse_loss(new_values, actor_cost.detach())
            # critic_clipped = F.mse_loss(
            #     old_values + torch.clamp(new_values - old_values, -clip_param, clip_param), 
            #     actor_cost.detach()
            # )
            # critic_loss = torch.max(critic_unclipped, critic_clipped)

            # Total loss
            total_loss = actor_loss - 0.01 * entropy.sum(dim=1).mean()

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.gradient_check:
                log_gradients(self.model)
            self.gradient_check = False
            self.optimizer.step()

            # Logging
            with torch.no_grad():
                clip_fraction = ((ratios - 1.0).abs() > clip_param).float().mean()
            
            logger['actor_loss'].append(actor_loss.item())
            # logger['critic_loss'].append(critic_loss.item())
            logger['advantage'].append(advantage.mean().item())
            logger['ratios'].append(ratios.mean().item())
            logger['entropy'].append(entropy.mean().item())
            logger['clip_fraction'].append(clip_fraction.item())

