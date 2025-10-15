import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm

from model.actor_network import SinkhornPermutationActor
from model.critic_network import Critic
from trainer.memory import Memory
from utils.utils import compute_euclidean_tour, get_initial_tours
from utils.logger import log_gradients


class SPGTrainer:
    def __init__(self, opts) -> None:
        self.opts = opts

        # Initialize the actor with optimizer and learning rate scheduler
        if opts.actor_load_path:
            print(f"Loading actor model from {opts.actor_load_path}")
            self.actor = torch.load(opts.actor_load_path, map_location=opts.device)
        else:
            self.actor = SinkhornPermutationActor(
                input_dim = opts.input_dim,
                embedding_dim = opts.embedding_dim,
                kNN_neighbors = opts.kNN_neighbors,
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

        # Initialize the critic with optimizer, learning rate scheduler and loss function
        if opts.critic_load_path:
            print(f"Loading critic model from {opts.critic_load_path}")
            self.critic = torch.load(opts.critic_load_path, map_location=opts.device)
        else:
            self.critic = Critic(
                input_dim = opts.input_dim,
                embedding_dim = opts.embedding_dim,
                kNN_neighbors = opts.kNN_neighbors,
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
        # prepare datasets
        val_dataset = problem.make_dataset(
            size=self.opts.graph_size, num_samples=self.opts.eval_size, filename=self.opts.val_dataset)
        train_dataset = problem.make_dataset(
            size=self.opts.graph_size, num_samples=self.opts.problem_size)

        # Initialize the memory buffer for experience replay
        replay_buffer = Memory(
            limit = self.opts.buffer_size,
            action_shape = (self.opts.graph_size, self.opts.graph_size),
            observation_shape = (self.opts.graph_size, self.opts.input_dim), 
            device = self.opts.device
        )

        # start training loop
        for epoch in range(self.opts.n_epochs):
            print(f"\nTraining Epoch {epoch}:")
            print(f"-  Replay Buffer Size: {len(replay_buffer)}")
            print(f"-  Actor Learning Rate: {self.actor_optimizer.param_groups[0]['lr']:.6f}")
            print(f"-  Actor Sinkhorn Temperature: {self.actor.decoder.gs_tau:.6f}")
            print(f"-  Critic Learning Rate: {self.critic_optimizer.param_groups[0]['lr']:.6f}")

            logger = {
                'baseline_cost': [],
                'actual_cost': [],
                'reward': [],
                'actor_loss': [],
                'critic_loss': []
            }

            start_time = time.time()

            self.train()
            training_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.opts.batch_size,
            )
            self.gradient_check = True

            for _, batch in enumerate(tqdm(training_dataloader, disable=self.opts.no_progress_bar)):
                self.train_batch(batch, replay_buffer, logger)

            epoch_duration = time.time() - start_time
            print(f"Training Epoch {epoch} completed. Results:")
            print(f"-  Epoch Runtime: {epoch_duration:.2f}s")
            print(f"-  Average Baseline Cost: {sum(logger['baseline_cost'])/len(logger['baseline_cost']):.4f}")
            print(f"-  Average Actual Cost: {sum(logger['actual_cost'])/len(logger['actual_cost']):.4f}")
            print(f"-  Average Reward: {sum(logger['reward'])/len(logger['reward']):.4f}")
            print(f"-  Average Actor Loss: {sum(logger['actor_loss'])/len(logger['actor_loss']):.4f}")
            print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")

            # update learning rate and sinkhorn temperature
            self.actor.decoder.gs_tau *= self.opts.sinkhorn_tau_decay
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
                torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
                torch.save(self.critic, f"{self.opts.save_dir}/critic_{self.opts.problem}_epoch{epoch + 1}.pt")
                print(f"Saved actor and critic models at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, replay_buffer: Memory, logger: dict):

        # --- ON-POLICY: Collect Experience ---
        ## get observations (initial tours) through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = batch['coordinates']

        baseline_tours = get_initial_tours(observation, self.opts.tour_heuristic)
        baseline_cost = compute_euclidean_tour(baseline_tours)

        ## Actor forward pass & tour construction & reward calculation
        dense_actions, discrete_actions = self.actor(observation)

        ## Print one sample for soft and hard permutation matrices to verify sinkhorn output
        # if self.gradient_check:
        #     print("Sample permutation matrices:")
        #     print(dense_actions[0])
        #     print(discrete_actions[0])

        ## Epsilon-greedy exploration - perform swap in 2-opt-style to the current tour
        if self.opts.epsilon > 0:
            B, N, _ = discrete_actions.shape
            device = discrete_actions.device

            # Decide for each problem in batch individually if to perform exploration
            explore_mask = torch.rand(B, device=device) < self.opts.epsilon
            P = explore_mask.sum()  # number of problems to explore
            if P > 0:
                batch_idxs = torch.where(explore_mask)[0]  # indices of problems to explore

                # select two random nodes for each selected problem in the batch
                swap_nodes = torch.multinomial(torch.ones(P, N, device=device), num_samples=2, replacement=False)
                i, j = swap_nodes[:, 0], swap_nodes[:, 1]

                cols_i = discrete_actions[batch_idxs, :, i]
                cols_j = discrete_actions[batch_idxs, :, j]
                discrete_actions[batch_idxs, :, i] = cols_j
                discrete_actions[batch_idxs, :, j] = cols_i

                dense_cols_i = dense_actions[batch_idxs, :, i]
                dense_cols_j = dense_actions[batch_idxs, :, j]
                dense_actions[batch_idxs, :, i] = dense_cols_j
                dense_actions[batch_idxs, :, j] = dense_cols_i

        ## Reward calculation using soft actions (tour distributions)
        actual_cost = compute_euclidean_tour(torch.bmm(discrete_actions.transpose(1, 2), observation))
        reward = (baseline_cost - actual_cost) * self.opts.reward_scale  # Apply reward scaling

        ## Add experience to replay buffer & log statistics
        replay_buffer.append(
            observations = observation.detach(), 
            discrete_actions = discrete_actions.detach(), 
            dense_actions = dense_actions.detach(), 
            rewards = reward.detach()
        )
        logger['baseline_cost'].append(baseline_cost.mean().item())
        logger['actual_cost'].append(actual_cost.mean().item())
        logger['reward'].append(reward.mean().item())

        # --- Off-Policy: Network Updates with Experience Replay ---
        ## Do not proceed with training if the buffer is not full enough
        if len(replay_buffer) < self.opts.batch_size:
            return
        sampled_obs, sampled_disc_actions, sampled_dense_actions, sampled_rewards = replay_buffer.sample(self.opts.batch_size)

        # Critic Update - compute Q(s, a) from hard and soft actions
        # this is to ensure connection to the real environment (via hard) and 
        # to provide a smooth gradient signal (via soft)
        self.critic_optimizer.zero_grad()

        hard_Q = self.critic(sampled_obs, sampled_disc_actions)
        soft_Q = self.critic(sampled_obs, sampled_dense_actions)
        critic_loss = F.mse_loss(hard_Q, sampled_rewards) + F.mse_loss(soft_Q, hard_Q.detach())
        logger['critic_loss'].append(critic_loss.item())

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        if self.gradient_check:
            log_gradients(self.critic)
        self.critic_optimizer.step()

        # Actor Update - compute policy gradient loss using the soft actions
        # for improved gradient signal. Generate new actions for the sampled observations.
        # Actor loss is computed as negative Q value to maximize expected reward.
        self.actor_optimizer.zero_grad()

        new_dense_actions, _ = self.actor(sampled_obs)
        actor_loss = -self.critic(sampled_obs, new_dense_actions).mean()
        logger['actor_loss'].append(actor_loss.item())

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
        initial_tours = get_initial_tours(coords, self.opts.tour_heuristic)
        initial_tour_lengths = compute_euclidean_tour(initial_tours)

        # Actor forward pass & tour construction & reward calculation
        dense_actions, discrete_actions = self.actor(initial_tours)

        # Reward calculation, TODO: Include expected reward for soft actions?
        new_tour_lengths = compute_euclidean_tour(torch.bmm(discrete_actions.transpose(1, 2), initial_tours))
        reward = (initial_tour_lengths - new_tour_lengths) * self.opts.reward_scale  # Apply reward scaling
