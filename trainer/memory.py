import torch


class Memory:
    def __init__(self, limit: int, action_shape: tuple, observation_shape: tuple, device: torch.device):
        """
        A vectorized, efficient replay buffer that lives on the GPU. It stores experiences and allows for random sampling of batches.
        Buffer stores observations, discrete actions, dense actions, and rewards.
        Args:
            limit: int - maximum number of experiences to store in the buffer
            action_shape: tuple - shape of the action space
            observation_shape: tuple - shape of the observation space
            device: torch.device - device to store the buffer on (e.g., 'cuda' or 'cpu')
        """
        self.limit = limit
        self.device = device

        # Pre-allocate memory on the target device
        self.observations = torch.zeros((limit, *observation_shape), device=self.device, dtype=torch.float32)
        self.discrete_actions = torch.zeros((limit, *action_shape), device=self.device, dtype=torch.bool)
        self.dense_actions = torch.zeros((limit, *action_shape), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros((limit, 1), device=self.device, dtype=torch.float32)
        
        self.position = 0
        self.size = 0

    def append(self, observations: torch.Tensor, discrete_actions: torch.Tensor, dense_actions: torch.Tensor, rewards: torch.Tensor):
        """
        Appends a batch of experiences to the buffer in a vectorized manner.
        """
        batch_size = observations.shape[0]
        
        # Generate indices for insertion using modulo arithmetic
        indices = torch.arange(self.position, self.position + batch_size, device=self.device) % self.limit
        
        # Vectorized assignment - one large operation per tensor
        self.observations[indices] = observations
        self.discrete_actions[indices] = discrete_actions
        self.dense_actions[indices] = dense_actions
        self.rewards[indices] = rewards.unsqueeze(1) # Ensure rewards is (B, 1)
        
        # Update position and size
        self.position = (self.position + batch_size) % self.limit
        self.size = min(self.size + batch_size, self.limit)

    def sample(self, batch_size: int):
        """
        Samples a random batch of experiences from the buffer.
        All operations are performed on the GPU.
        """
        # Generate random indices directly on the GPU
        batch_idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        obs_batch = self.observations[batch_idxs]
        discrete_actions_batch = self.discrete_actions[batch_idxs]
        dense_actions_batch = self.dense_actions[batch_idxs]
        reward_batch = self.rewards[batch_idxs]
        
        return obs_batch, discrete_actions_batch, dense_actions_batch, reward_batch

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
    memory = Memory(limit=100000, action_shape=(10, 10), observation_shape=(10, 2), device=device)

    # Create sample data on the correct device
    batch_size = 128
    states = torch.randn(batch_size, 10, 2, device=device)
    discrete_actions = torch.ones(batch_size, 10, 10, device=device, dtype=torch.bool)
    dense_actions = torch.rand(batch_size, 10, 10, device=device)
    rewards = torch.randn(batch_size, device=device)
    
    # Append data
    memory.append(states, discrete_actions, dense_actions, rewards)
    print(f"Appended {batch_size} items. Buffer size: {len(memory)}")
    
    # Sample data
    sample_size = 32
    s_batch, psi_batch, a_batch, r_batch = memory.sample(sample_size)
    
    print(f"\nSampled a batch of {sample_size}:")
    print(f"  States shape: {s_batch.shape}, Device: {s_batch.device}")
    print(f"  Discrete Actions shape: {psi_batch.shape}, Device: {psi_batch.device}")
    print(f"  Dense Actions shape: {a_batch.shape}, Device: {a_batch.device}")
    print(f"  Rewards shape: {r_batch.shape}, Device: {r_batch.device}")