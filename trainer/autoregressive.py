import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from tqdm import tqdm

from model.actor_network import ARPointerActor
from model.critic_network import Critic
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


    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()


    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()


    def start_training(self, problem):
        # prepare datasets
        train_dataset = problem.make_dataset(
            size=self.opts.graph_size, num_samples=self.opts.problem_size)

        # start training loop
        for epoch in range(self.opts.n_epochs):
            print(f"\nTraining Epoch {epoch}:")
            print(f"-  Actor Learning Rate: {self.actor_optimizer.param_groups[0]['lr']:.6f}")

            logger = {
                'baseline_tour_length': [],
                'actual_tour_length': [],
                'advantage': [],
                'action_log_prob': [],
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
                self.train_batch(batch, logger)

            epoch_duration = time.time() - start_time
            print(f"Training Epoch {epoch} completed. Results:")
            print(f"-  Epoch Runtime: {epoch_duration:.2f}s")
            print(f"-  Average Baseline Tour Length: {sum(logger['baseline_tour_length'])/len(logger['baseline_tour_length']):.4f}")
            print(f"-  Average Actual Tour Length: {sum(logger['actual_tour_length'])/len(logger['actual_tour_length']):.4f}")
            print(f"-  Average Advantage: {sum(logger['advantage'])/len(logger['advantage']):.4f}")
            print(f"-  Average Action Log Probability: {sum(logger['action_log_prob'])/len(logger['action_log_prob']):.4f}")
            print(f"-  Average Actor Loss: {sum(logger['actor_loss'])/len(logger['actor_loss']):.4f}")

            # update learning rate
            self.actor_scheduler.step()

            if (self.opts.checkpoint_epochs != 0 and epoch % self.opts.checkpoint_epochs == 0) or epoch == self.opts.n_epochs - 1:
                torch.save(self.actor, f"{self.opts.save_dir}/actor_{self.opts.problem}_epoch{epoch + 1}.pt")
                print(f"Saved model at epoch {epoch + 1} to {self.opts.save_dir}")


    def train_batch(self, batch: dict, logger: dict):

        # get observations (initial tours) through heuristic from the environment
        batch = {k: v.to(self.opts.device) for k, v in batch.items()}
        observation = batch['coordinates']

        # Actor forward pass - TODO: Evaluate if epsilon-greedy 2-opt exploration is beneficial
        actions, prob_dist = self.actor(observation)
        log_prob_sums = (torch.log(prob_dist + 1e-9) * actions.detach()).sum(dim=(1, 2))  # (B,)
        actor_tour_lengths = compute_euclidean_tour(torch.bmm(actions.transpose(1, 2), observation))  # (B,)

        # Calculate actor loss with baseline
        baseline_tours = get_heuristic_tours(observation, self.opts.tour_heuristic)
        baseline_tour_lengths = compute_euclidean_tour(baseline_tours)
        advantage = baseline_tour_lengths - actor_tour_lengths
        actor_loss = -(advantage.detach() * log_prob_sums).mean()

        # Logging
        logger['baseline_tour_length'].append(baseline_tour_lengths.mean().item())
        logger['actual_tour_length'].append(actor_tour_lengths.mean().item())
        logger['advantage'].append(advantage.mean().item())
        logger['action_log_prob'].append(log_prob_sums.mean().item())
        logger['actor_loss'].append(actor_loss.item())

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        if self.gradient_check:
            log_gradients(self.actor)
        self.actor_optimizer.step()
        self.gradient_check = False


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
