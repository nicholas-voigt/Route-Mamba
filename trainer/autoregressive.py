import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm

from model.actor_network import ARPointerActor
from model.critic_network import Critic
from model import components as mc
from utils.utils import compute_euclidean_tour, get_heuristic_tours
from utils.logger import log_gradients


class ARTrainer:
    def __init__(self, opts) -> None:
        self.opts = opts

        # Initialize the actor with optimizer and learning rate scheduler
        if opts.actor_load_path:
            print(f"Loading actor model from {opts.actor_load_path}")
            self.actor = torch.load(opts.actor_load_path, map_location=opts.device)
        else:
            self.actor = ARPointerActor(
                input_dim = opts.input_dim,
                embedding_dim = opts.embedding_dim,
                mamba_hidden_dim = opts.mamba_hidden_dim,
                mamba_layers = opts.mamba_layers,
                dropout = opts.dropout
            ).to(opts.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=opts.actor_lr)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lambda epoch: opts.actor_lr_decay ** epoch)

        # Initialize the critic with optimizer and learning rate scheduler
        # if opts.critic_load_path:
        #     print(f"Loading critic model from {opts.critic_load_path}")
        #     self.critic = torch.load(opts.critic_load_path, map_location=opts.device)
        # else:
        #     self.critic = Critic(
        #         input_dim = opts.input_dim,
        #         embedding_dim = opts.embedding_dim,
        #         mamba_hidden_dim = opts.mamba_hidden_dim,
        #         mamba_layers = opts.mamba_layers,
        #         dropout = opts.dropout,
        #         mlp_ff_dim = opts.mlp_ff_dim,
        #         mlp_embedding_dim = opts.mlp_embedding_dim
        #     ).to(opts.device)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=opts.critic_lr)
        # self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lambda epoch: opts.critic_lr_decay ** epoch)


    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        # if not self.opts.eval_only: self.critic.train()


    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        # if not self.opts.eval_only: self.critic.eval()


    def start_training(self, problem):
        self.gradient_check = False
        
        # --- Critic Warm-up Phase ---
        # if self.opts.critic_warmup_epochs > 0:
        #     print(f"\nCritic Warm-up for {self.opts.critic_warmup_epochs} epochs ---")
        #     for epoch in range(self.opts.critic_warmup_epochs):
        #         # prepare training dataset
        #         train_dataset = problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.problem_size)
        #         training_dataloader = DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size)
                
        #         # Logging
        #         print(f"\nCritic Warm-up Epoch {epoch}:")
        #         print(f"-  Critic Learning Rate: {self.critic_optimizer.param_groups[0]['lr']:.6f}")
                
        #         logger = {
        #             'critic_cost': [],
        #             'critic_loss': []
        #         }

        #         # training batch loop
        #         self.train()
        #         start_time = time.time()

        #         for _, batch in enumerate(tqdm(training_dataloader, disable=self.opts.no_progress_bar)):
        #             self.train_batch(batch, logger, warmup_mode=True)

        #         epoch_duration = time.time() - start_time
        #         print(f"Critic Warm-up Epoch {epoch + 1} completed. Results:")
        #         print(f"-  Epoch Runtime: {epoch_duration:.2f}s")
        #         print(f"-  Average Critic Cost: {sum(logger['critic_cost'])/len(logger['critic_cost']):.4f}")
        #         print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")
        #         self.critic_scheduler.step()

        #     print("\n--- Critic Warm-up Finished ---\n")

        # --- Main Training Phase ---
        for epoch in range(self.opts.n_epochs):
            # prepare training dataset
            train_dataset = problem.make_dataset(size=self.opts.graph_size, num_samples=self.opts.problem_size)
            training_dataloader = DataLoader(dataset=train_dataset, batch_size=self.opts.batch_size)

            # Logging
            print(f"\nTraining Epoch {epoch}:")
            print(f"-  Actor Learning Rate: {self.actor_optimizer.param_groups[0]['lr']:.6f}")
            # print(f"-  Critic Learning Rate: {self.critic_optimizer.param_groups[0]['lr']:.6f}")

            logger = {
                'critic_cost': [],
                'actual_cost': [],
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
            # print(f"-  Average Critic Cost: {sum(logger['critic_cost'])/len(logger['critic_cost']):.4f}")
            print(f"-  Average Actual Cost: {sum(logger['actual_cost'])/len(logger['actual_cost']):.4f}")
            print(f"-  Average Actor Loss: {sum(logger['actor_loss'])/len(logger['actor_loss']):.4f}")
            # print(f"-  Average Critic Loss: {sum(logger['critic_loss'])/len(logger['critic_loss']):.4f}")

            # update learning rates
            self.actor_scheduler.step()
            # self.critic_scheduler.step()

            # if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
            #     torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
            #     torch.save(self.critic, f"{self.opts.save_dir}/critic_{self.opts.problem}_epoch{epoch + 1}.pt")
            #     print(f"Saved actor and critic models at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, logger: dict, warmup_mode: bool = False):
        # get observations (initial tours) through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = get_heuristic_tours(batch['coordinates'], self.opts.initial_tours)

        # Actor forward pass
        log_prob_matrix, tour_matrix = self.actor(observation)

        # Calculate actual cost of the tours generated by the actor
        actual_cost = compute_euclidean_tour(torch.bmm(tour_matrix.transpose(1, 2), observation))

        # Using greedy baseline
        baseline_tours = get_heuristic_tours(batch['coordinates'], self.opts.baseline_tours)
        estimated_cost = compute_euclidean_tour(baseline_tours)

        # # Critic value estimation  & loss calculation using MSE loss
        # estimated_cost = self.critic(observation).squeeze(1)
        # critic_loss = F.mse_loss(estimated_cost, actual_cost.detach())

        # Critic Logging
        logger['critic_cost'].append(estimated_cost.mean().item())
        # logger['critic_loss'].append(critic_loss.item())

        # # Critic Update
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        # if self.gradient_check:
        #     log_gradients(self.critic)
        # self.critic_optimizer.step()

        # if warmup_mode:
        #     return  # Skip actor update during warm-up phase

        # Actor loss using REINFORCE with critic baseline
        log_likelihood = -torch.sum(log_prob_matrix * tour_matrix, dim=(1, 2))
        advantage = ((actual_cost - estimated_cost) / estimated_cost).detach() * self.opts.reward_scale  # Apply reward scaling
        actor_loss = (advantage * log_likelihood).mean()

        # Logging
        logger['actual_cost'].append(actual_cost.mean().item())
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
