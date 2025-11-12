import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm

import components as mc
from utils.utils import compute_euclidean_tour, get_heuristic_tours, check_feasibility
from utils.logger import log_gradients


class Actor(nn.Module):
    def __init__(self, input_dim, embedding_dim, mamba_hidden_dim, mamba_layers, 
                 num_attention_heads, ffn_expansion, initial_identity_bias, gs_tau, gs_iters, method, dropout):
        super(Actor, self).__init__()

        # Model components
        self.feature_embedder = mc.StructuralEmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim
        )
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.encoder = mc.BidirectionalMambaEncoder(
            mamba_model_size = embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.encoder_norm = nn.LayerNorm(2 * embedding_dim)
        self.score_constructor = mc.AttentionScoreHead(
            model_dim = 2 * embedding_dim,
            num_heads = num_attention_heads,
            ffn_expansion = ffn_expansion,
            dropout = dropout
        )
        self.identity_bias = nn.Parameter(torch.tensor(initial_identity_bias, dtype=torch.float32), requires_grad=True)
        self.decoder = mc.GumbelSinkhornDecoder(
            gs_tau = gs_tau,
            gs_iters = gs_iters
        )
        self.tour_constructor = mc.TourConstructor(
            method = method
        )

    def forward(self, batch):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
        Returns:
            st_perm: (B, N, N) - doubly stochastic matrix (soft permutation matrix)
            hard_perm: (B, N, N) - permutation matrix (hard assignment of the tour)
        """
        # 1. Create Embeddings & normalize
        embeddings = self.feature_embedder(batch)  # (B, N, E)
        embeddings = self.embedding_norm(embeddings)  # (B, N, E)

        # 2. Encoder: Layered Mamba blocks with internal Pre-LN
        encoded_features = self.encoder(embeddings)
        encoded_features = self.encoder_norm(encoded_features)   # (B, N, 2E)

        # 3. Score Construction: Multi-Head Attention with FFN and internal Pre-LN and projection to scores
        score_matrix = self.score_constructor(encoded_features)  # (B, N, N)
        score_matrix = F.layer_norm(score_matrix, score_matrix.shape[1:], eps=1e-5)
        # Identity Bias to encourage near-identity initial permutations
        identity_matrix = torch.eye(score_matrix.size(1), device=score_matrix.device) * self.identity_bias
        biased_score_matrix = score_matrix + identity_matrix

        # 4. Decoder Workshop: Use Gumbel-Sinkhorn to get soft permutation matrix & hard assignment via tour construction
        soft_perm = self.decoder(biased_score_matrix)  # (B, N, N)
        hard_perm = self.tour_constructor(soft_perm)

        return soft_perm, hard_perm


class Critic(nn.Module):
    def __init__(self, input_dim, embedding_dim, mamba_hidden_dim, mamba_layers,
                 dropout, mlp_ff_dim, mlp_embedding_dim):
        super(Critic, self).__init__()

        # State Encoder
        self.state_embedder = mc.StructuralEmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim
        )
        self.state_embedding_norm = nn.LayerNorm(embedding_dim)
        self.state_encoder = mc.BidirectionalMambaEncoder(
            mamba_model_size = embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.state_encoder_norm = nn.LayerNorm(2 * embedding_dim)

        # Value Decoder
        self.value_decoder = mc.MLP(
            input_dim = 2 * embedding_dim,
            feed_forward_dim = mlp_ff_dim,
            embedding_dim = mlp_embedding_dim,
            dropout = dropout,
            output_dim = 1
        )

    def forward(self, state):
        """
        Forward pass of the Critic network.
        Args:
            state: Tensor of shape (B, N, 2) representing the coordinates of the nodes.
        Returns:
            value: Tensor of shape (B, 1) representing the estimated Q-value for the (state, action) pair.
        """
        # Encode the State
        state_embedding = self.state_embedder(state)  # (B, N, E)
        state_embedding = self.state_embedding_norm(state_embedding) # (B, N, E)

        state_encoding = self.state_encoder(state_embedding)  # (B, N, 2E)
        state_encoding = self.state_encoder_norm(state_encoding)  # (B, N, 2E)

        # Decode Q-Value
        q = self.value_decoder(state_encoding.mean(dim=1))  # (B, 1)
        return q


class PPPOTrainer:
    def __init__(self, opts) -> None:
        self.opts = opts

        # Initialize the actor with optimizer and learning rate scheduler
        if opts.actor_load_path:
            print(f"Loading actor model from {opts.actor_load_path}")
            self.actor = torch.load(opts.actor_load_path, map_location=opts.device)
        else:
            self.actor = Actor(
                input_dim = opts.input_dim,
                embedding_dim = opts.embedding_dim,
                mamba_hidden_dim = opts.mamba_hidden_dim,
                mamba_layers = opts.mamba_layers,
                num_attention_heads = opts.num_attention_heads,
                ffn_expansion = opts.ffn_expansion,
                initial_identity_bias = opts.initial_identity_bias,
                gs_tau = opts.sinkhorn_tau,
                gs_iters = opts.sinkhorn_iters,
                method = opts.tour_method,
                dropout = opts.dropout
            ).to(opts.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=opts.actor_lr)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lambda epoch: opts.actor_lr_decay ** epoch)

        # Initialize the critic with optimizer and learning rate scheduler
        if opts.critic_load_path:
            print(f"Loading critic model from {opts.critic_load_path}")
            self.critic = torch.load(opts.critic_load_path, map_location=opts.device)
        else:
            self.critic = Critic(
                input_dim = opts.input_dim,
                embedding_dim = opts.embedding_dim,
                mamba_hidden_dim = opts.mamba_hidden_dim,
                mamba_layers = opts.mamba_layers,
                dropout = opts.dropout,
                mlp_ff_dim = opts.mlp_ff_dim,
                mlp_embedding_dim = opts.mlp_embedding_dim
            ).to(opts.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=opts.critic_lr)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lambda epoch: opts.critic_lr_decay ** epoch)


    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.opts.eval_only: self.critic.train()        


    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.opts.eval_only: self.critic.eval()


    def start_training(self, problem):
        self.gradient_check = False
        
        # --- Critic Warm-up Phase ---
        if self.opts.critic_warmup_epochs > 0:
            print(f"\nCritic Warm-up for {self.opts.critic_warmup_epochs} epochs ---")
            for epoch in range(self.opts.critic_warmup_epochs):
                # prepare training dataset
                train_dataset = problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.problem_size)
                training_dataloader = DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size)
                
                # Logging
                print(f"\nCritic Warm-up Epoch {epoch}:")
                print(f"-  Critic Learning Rate: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
                
                logger = {
                    'tour_cost': [],
                    'baseline_cost': [],
                    'critic_cost': [],
                    'critic_loss': [],
                    'advantage': [],
                }

                # training batch loop
                self.train()
                start_time = time.time()

                for _, batch in enumerate(tqdm(training_dataloader, disable=self.opts.no_progress_bar)):
                    self.train_batch(batch, logger, warmup_mode=True)

                epoch_duration = time.time() - start_time
                print(f"Critic Warm-up Epoch {epoch + 1} completed. Results:")
                print(f"-  Epoch Runtime: {epoch_duration:.2f}s")
                print(f"-  Average Tour Cost: {sum(logger['tour_cost'])/len(logger['tour_cost']):.4f}")
                print(f"-  Average Baseline Cost: {sum(logger['baseline_cost'])/len(logger['baseline_cost']):.4f}")
                print(f"-  Average Critic Cost: {sum(logger['critic_cost'])/len(logger['critic_cost']):.4f}")
                print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")
                print(f"-  Average Advantage: {sum(logger['advantage'])/len(logger['advantage']):.4f}")
                self.critic_scheduler.step()

            print("\n--- Critic Warm-up Finished ---\n")

        # --- Main Training Phase ---
        for epoch in range(self.opts.n_epochs):
            # prepare training dataset
            train_dataset = problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.problem_size)
            training_dataloader = DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size)

            # Logging
            print(f"\nTraining Epoch {epoch}:")
            print(f"-  Actor Learning Rate: {self.actor_optimizer.param_groups[0]['lr']:.6f}")
            print(f"-  Critic Learning Rate: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
            print(f"-  Actor Sinkhorn Temperature: {self.actor.decoder.gs_tau:.6f}")

            logger = {
                'tour_cost': [],
                'baseline_cost': [],
                'critic_cost': [],
                'actor_loss': [],
                'critic_loss': [],
                'advantage': [],
                'ratios': [],
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
            print(f"-  Average Critic Cost: {sum(logger['critic_cost'])/len(logger['critic_cost']):.4f}")
            print(f"-  Average Actor Loss: {sum(logger['actor_loss'])/len(logger['actor_loss']):.4f}")
            print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")
            print(f"-  Average Advantage: {sum(logger['advantage'])/len(logger['advantage']):.4f}")
            print(f"-  Average Policy Ratios: {sum(logger['ratios'])/len(logger['ratios']):.4f}")
            print(f"-  Clip Fraction: {sum(logger['clip_fraction'])/len(logger['clip_fraction']):.4f}")

            # update learning rate and sinkhorn temperature
            self.actor.decoder.gs_tau = max(self.actor.decoder.gs_tau * self.opts.sinkhorn_tau_decay, 1.0)
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
                torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
                torch.save(self.critic, f"{self.opts.save_dir}/critic_{self.opts.problem}_epoch{epoch + 1}.pt")
                print(f"Saved actor and critic models at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, logger: dict, warmup_mode: bool = False):

        # --- 1. Collect Trajectories ---
        # get observations (Polar-ordered coordinates (canonical representation)) from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours) # (B, N, 2)

        # Forward pass
        with torch.no_grad():
            # Actor forward pass
            log_probs, action = self.actor(observation)
            tours = torch.bmm(action.transpose(1, 2), observation)  # (B, N, 2)
            check_feasibility(observation, tours)
            actor_cost = compute_euclidean_tour(tours)

            # Critic value estimation
            values = self.critic(observation).squeeze(1)

            # Store "old" policy & values
            old_lp_sum = torch.sum(log_probs * action, dim=(1, 2)).detach()  # log-probability sum of the actions taken
            old_values = values.detach()

        # --- 2. Calculate Advantage with GAE ---
        # Heuristic baseline (greedy) (as reference only)
        baseline_tours = get_heuristic_tours(observation, self.opts.baseline_tours)
        baseline_cost = compute_euclidean_tour(baseline_tours)

        # Normalized advantage over critic estimation
        raw_advantage = (values - actor_cost) / (values + 1e-6)  # (B,)
        advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-6)  # Normalize advantage

        # --- 3. PPO Update (multiple epochs over collected data) ---
        ppo_metrics = {'actor_loss': [], 'critic_loss': [], 'ratios': [], 'clip_fraction': []}

        for _ in range(2):
            clip_param = 0.5

            # Critic evaluation
            new_values = self.critic(observation).squeeze(1)
            critic_unclipped = F.mse_loss(new_values, actor_cost.detach())
            critic_clipped = F.mse_loss(
                old_values + torch.clamp(new_values - old_values, -clip_param, clip_param), 
                actor_cost.detach()
            )
            critic_loss = torch.max(critic_unclipped, critic_clipped)

            # Backpropagation
            self.critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            if self.gradient_check:
                log_gradients(self.critic)
            self.critic_optimizer.step()
            ppo_metrics['critic_loss'].append(critic_loss.item())

            if warmup_mode:
                continue  # during warm-up phase, only train the critic
            
            # Actor evaluation and update
            # Re-evaluate actions with current policy
            log_probs, _ = self.actor(observation, actions=action)
            new_lp_sum = torch.sum(log_probs * action, dim=(1, 2))  # log-probability sum of the actions taken
            ratios = torch.exp(new_lp_sum - old_lp_sum)  # (B,)
            actor_unclipped = -advantage * ratios
            actor_clipped = -advantage * torch.clamp(ratios, 1 - clip_param, 1 + clip_param)
            actor_loss = torch.max(actor_unclipped, actor_clipped).mean()

            # Backpropagation
            self.actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            if self.gradient_check:
                log_gradients(self.actor)
            self.gradient_check = False
            self.actor_optimizer.step()

            # Logging
            with torch.no_grad():
                clip_fraction = ((ratios - 1.0).abs() > clip_param).float().mean()
            ppo_metrics['actor_loss'].append(actor_loss.item())
            ppo_metrics['ratios'].append(ratios.mean().item())
            ppo_metrics['clip_fraction'].append(clip_fraction.item())

        # Final logging after PPO epochs
        logger['tour_cost'].append(actor_cost.mean().item())
        logger['baseline_cost'].append(baseline_cost.mean().item())
        logger['critic_cost'].append(values.mean().item())
        logger['advantage'].append(advantage.mean().item())
        logger['critic_loss'].append(sum(ppo_metrics['critic_loss']) / len(ppo_metrics['critic_loss']))
        if not warmup_mode:
            logger['actor_loss'].append(sum(ppo_metrics['actor_loss']) / len(ppo_metrics['actor_loss']))
            logger['ratios'].append(sum(ppo_metrics['ratios']) / len(ppo_metrics['ratios']))
            logger['clip_fraction'].append(sum(ppo_metrics['clip_fraction']) / len(ppo_metrics['clip_fraction']))


    def start_evaluation(self, problem):
        # prepare dataset
        val_dataset = problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.eval_size, filename=self.opts.val_dataset)
        dataloader = DataLoader(val_dataset, batch_size=self.opts.batch_size)

        # Logging
        print("\nStarting Evaluation:")
        print(f"-  Evaluating {self.opts.problem}-{self.opts.graph_size}")
        print(f"-  Eval Dataset Size: {len(val_dataset)}")
        print(f"-  Batch Size: {self.opts.batch_size}")

        logger = {
            'actual_cost': []
        }

        # start evaluation loop
        self.eval()
        start_time = time.time()
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, disable=self.opts.no_progress_bar)):
                self.evaluate_batch(batch, logger)

        eval_duration = time.time() - start_time
        print(f"Evaluation completed. Results:")
        print(f"-  Runtime: {eval_duration:.2f}s")
        print(f"-  Average Cost: {sum(logger['actual_cost'])/len(logger['actual_cost']):.4f}")

        # Precise logging of evaluation results
        print("\n--- Batch-Specific Evaluation Results ---")
        for i, cost in enumerate(logger['actual_cost']):
            print(f"Batch {i+1}: Average Cost: {cost:.4f}")


    def evaluate_batch(self, batch: dict, logger: dict):
        # get observations through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours)

        # Actor forward pass to generate discrete actions (tour permutations) and probabilistic actions (tour distributions)
        _, action = self.actor(observation)

        # Calculate actual cost of the tours generated by the actor
        actual_cost = compute_euclidean_tour(torch.bmm(action.transpose(1, 2), observation))
        logger['actual_cost'].append(actual_cost.mean().item())