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


    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()


    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()


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

            logger = {
                'baseline_cost': [],
                'actual_cost': [],
                'reward': [],
                'actor_loss': []
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

            # update learning rate and sinkhorn temperature
            self.actor.decoder.gs_tau *= self.opts.sinkhorn_tau_decay
            self.actor_scheduler.step()

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
                torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
                print(f"Saved actor and critic models at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, replay_buffer: Memory, logger: dict):

        # --- ON-POLICY: Collect Experience ---
        ## get observations (initial tours) through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = batch['coordinates']

        baseline_tours = get_initial_tours(observation, self.opts.tour_heuristic)
        baseline_cost = compute_euclidean_tour(baseline_tours)

        ## Actor forward pass & tour construction & reward calculation
        _, discrete_actions = self.actor(observation)

        ## Reward calculation using soft actions (tour distributions)
        actual_cost = compute_euclidean_tour(torch.bmm(discrete_actions.transpose(1, 2), observation))
        reward = (baseline_cost - actual_cost) * self.opts.reward_scale  # Apply reward scaling

        logger['baseline_cost'].append(baseline_cost.mean().item())
        logger['actual_cost'].append(actual_cost.mean().item())
        logger['reward'].append(reward.mean().item())

        # Actor Update - compute policy gradient loss using the soft Q values
        self.actor_optimizer.zero_grad()

        actor_loss = -reward.mean()
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
