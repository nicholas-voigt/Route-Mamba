import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm

from model.actor_network import SinkhornPermutationActor
from model.critic_network import Critic
from trainer.memory import Memory
from utils.utils import compute_euclidean_tour, get_heuristic_tours
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
                    'critic_cost': [],
                    'critic_loss': []
                }

                # training batch loop
                self.train()
                start_time = time.time()

                for _, batch in enumerate(tqdm(training_dataloader, disable=self.opts.no_progress_bar)):
                    self.train_batch(batch, logger, warmup_mode=True)

                epoch_duration = time.time() - start_time
                print(f"Critic Warm-up Epoch {epoch + 1} completed. Results:")
                print(f"-  Epoch Runtime: {epoch_duration:.2f}s")
                print(f"-  Average Critic Cost: {sum(logger['critic_cost'])/len(logger['critic_cost']):.4f}")
                print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")
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
                'critic_cost': [],
                'actual_cost': [],
                'entropy': [],
                'actor_loss': [],
                'critic_loss': []
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
            print(f"-  Average Critic Cost: {sum(logger['critic_cost'])/len(logger['critic_cost']):.4f}")
            print(f"-  Average Actual Cost: {sum(logger['actual_cost'])/len(logger['actual_cost']):.4f}")
            print(f"-  Average Entropy: {sum(logger['entropy'])/len(logger['entropy']):.4f}")
            print(f"-  Average Actor Loss: {sum(logger['actor_loss'])/len(logger['actor_loss']):.4f}")
            print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")

            # update learning rate and sinkhorn temperature
            self.actor.decoder.gs_tau = max(self.actor.decoder.gs_tau * self.opts.sinkhorn_tau_decay, 1.0)
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
                torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
                torch.save(self.critic, f"{self.opts.save_dir}/critic_{self.opts.problem}_epoch{epoch + 1}.pt")
                print(f"Saved actor and critic models at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, logger: dict, warmup_mode: bool = False):

        # --- ON-POLICY: Collect Experience ---
        ## get observations (initial tours) through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours)

        # Actor forward pass to generate discrete actions (tour permutations) and probabilistic actions (tour distributions)
        log_probs, action = self.actor(observation)

        ## Epsilon-greedy exploration - perform swap in 2-opt-style to the current tour
        if self.opts.epsilon > 0:
            B, N, _ = action.shape
            device = action.device

            # Decide for each problem in batch individually if to perform exploration
            explore_mask = torch.rand(B, device=device) < self.opts.epsilon
            P = explore_mask.sum()  # number of problems to explore
            if P > 0:
                batch_idxs = torch.where(explore_mask)[0]  # indices of problems to explore

                # Clone the action and probs tensors to avoid in-place operations on the original tensors
                action = action.clone()
                log_probs = log_probs.clone()

                # select two random nodes for each selected problem in the batch
                swap_nodes = torch.multinomial(torch.ones(P, N, device=device), num_samples=2, replacement=False)
                i, j = swap_nodes[:, 0], swap_nodes[:, 1]

                actions_i = action[batch_idxs, :, i].clone()
                actions_j = action[batch_idxs, :, j].clone()
                action[batch_idxs, :, i] = actions_j
                action[batch_idxs, :, j] = actions_i

                probs_i = log_probs[batch_idxs, :, i].clone()
                probs_j = log_probs[batch_idxs, :, j].clone()
                log_probs[batch_idxs, :, i] = probs_j
                log_probs[batch_idxs, :, j] = probs_i

        # Calculate actual cost of the tours generated by the actor
        actual_cost = compute_euclidean_tour(torch.bmm(action.transpose(1, 2), observation))

        # Critic value estimation  & loss calculation using MSE loss
        estimated_cost = self.critic(observation).squeeze(1)
        critic_loss = F.mse_loss(estimated_cost, actual_cost.detach())

        # Critic Logging
        logger['critic_cost'].append(estimated_cost.mean().item())
        logger['critic_loss'].append(critic_loss.item())

        # Critic Update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        if self.gradient_check:
            log_gradients(self.critic)
        self.critic_optimizer.step()

        if warmup_mode:
            return  # during warm-up phase, only train the critic
        
        # Actor loss using REINFORCE with critic baseline
        log_likelihood = -torch.sum(log_probs * action, dim=(1, 2))
        advantage = ((actual_cost - estimated_cost) / estimated_cost).detach() * self.opts.reward_scale  # Apply reward scaling
        actor_loss = (advantage * log_likelihood).mean()
        # Entropy regularization to encourage exploration (optional)
        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=(1, 2)).mean()
        # actor_loss = reinforce_loss - self.opts.lambda_auxiliary_loss * entropy

        # Logging
        logger['actual_cost'].append(actual_cost.mean().item())
        logger['entropy'].append(entropy.item())
        logger['actor_loss'].append(actor_loss.item())

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