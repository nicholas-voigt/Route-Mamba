import torch


class Memory:
    def __init__(self, limit: int, obs_shape: tuple, action_shape: tuple, reward_shape: tuple, device: torch.device):
        """
        A vectorized, efficient replay buffer that lives on the GPU. It stores experiences and allows for random sampling of batches.
        Buffer stores observations, discrete actions, dense actions, and rewards.
        Args:
            limit: int - maximum number of experiences to store in the buffer
            obs_shape: tuple - shape of the observation space
            action_shape: tuple - shape of the action space (also used for log_probs)
            reward_shape: tuple - shape of the reward space
            device: torch.device - device to store the buffer on (e.g., 'cuda' or 'cpu')
        """
        self.limit = limit
        self.device = device

        # Pre-allocate memory on the target device
        self.observations = torch.zeros((limit, *obs_shape), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((limit, *action_shape), device=self.device, dtype=torch.float32)
        self.log_probs = torch.zeros((limit, *action_shape), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros((limit, *reward_shape), device=self.device, dtype=torch.float32)
        
        self.position = 0
        self.size = 0


    def append(self, obs: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor, rewards: torch.Tensor):
        """
        Appends a batch of experiences to the buffer in a vectorized manner.
        """
        batch_size = obs.shape[0]
        
        # Generate indices for insertion using modulo arithmetic
        indices = torch.arange(self.position, self.position + batch_size, device=self.device) % self.limit
        
        # Vectorized assignment - one large operation per tensor
        self.observations[indices] = obs
        self.actions[indices] = actions
        self.log_probs[indices] = log_probs
        self.rewards[indices] = rewards
        
        # Update position and size
        self.position = (self.position + batch_size) % self.limit
        self.size = min(self.size + batch_size, self.limit)


    def sample(self, batch_size: int):
        """
        Samples a random batch of experiences from the buffer.
        All operations are performed on the specified device.
        """
        # Generate random indices
        batch_idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        obs = self.observations[batch_idxs]
        actions = self.actions[batch_idxs]
        log_probs = self.log_probs[batch_idxs]
        rewards = self.rewards[batch_idxs]
        
        return obs, actions, log_probs, rewards


    def __len__(self):
        return self.size


# exemplary usage
if __name__ == '__main__':
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, running example on CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Initialize memory on the correct device
    memory = Memory(limit=100000, obs_shape=(10, 2), action_shape=(10,), reward_shape=(1,), device=device)

    # Create sample data on the correct device
    batch_size = 64
    states = torch.randn(batch_size, 10, 2, device=device)
    discrete_actions = torch.ones(batch_size, 10, device=device)
    log_probabilities = torch.rand(batch_size, 10, device=device)
    rewards = torch.randn(batch_size, 1, device=device)
    
    # Append data
    memory.append(states, discrete_actions, log_probabilities, rewards)
    print(f"Appended {batch_size} items. Buffer size: {len(memory)}")
    
    # Sample data
    sample_size = 32
    sampled_obs, sampled_actions, sampled_logprobs, sampled_rewards = memory.sample(sample_size)
    
    print(f"\nSampled a batch of {sample_size}:")
    print(f"  States shape: {sampled_obs.shape}, Device: {sampled_obs.device}")
    print(f"  Discrete Actions shape: {sampled_actions.shape}, Device: {sampled_actions.device}")
    print(f"  Dense Actions shape: {sampled_logprobs.shape}, Device: {sampled_logprobs.device}")
    print(f"  Rewards shape: {sampled_rewards.shape}, Device: {sampled_rewards.device}")