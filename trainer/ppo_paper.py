import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import numpy as np
from mamba_ssm import Mamba


class TSPEnvironment:
    def __init__(self, num_cities=20):
        self.num_cities = num_cities
        self.cities = np.random.rand(num_cities, 2) * 100
        self.distance_matrix = self.compute_distance_matrix()
        self.reset()

    def compute_distance_matrix(self):
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                dist_matrix[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist_matrix

    def reset(self):
        self.visited = set()
        self.current_city = np.random.randint(0, self.num_cities)
        self.visited.add(self.current_city)
        self.available_cities = set(range(self.num_cities)) - self.visited
        self.total_distance = 0
        return self.get_normalized_state()  # Using normalized state

    def get_normalized_state(self):
        # Normalize city coordinates to [0,1]
        normalized_cities = self.cities / 100.0
        
        # Create visited vector
        visited_vector = np.array([1 if i in self.visited else 0 for i in range(self.num_cities)])
        
        # Add last 3 visited cities (normalized indices)
        last_visited = list(self.visited)[-3:] if len(self.visited) >= 3 else [-1, -1, -1]
        last_visited += [-1] * (3 - len(last_visited))
        normalized_last_visited = [x/self.num_cities if x != -1 else 0 for x in last_visited]
        
        return np.concatenate((normalized_cities.flatten(), visited_vector, normalized_last_visited)).tolist()

    def step(self, action):
        if action not in self.available_cities:
            return self.get_normalized_state(), -1.0, False  # Normalized penalty

        distance = self.distance_matrix[self.current_city, action]
        self.total_distance += distance
        self.current_city = action
        self.visited.add(action)
        self.available_cities.remove(action)

        done = len(self.visited) == self.num_cities

        # Normalized reward between [0,1]
        reward = -distance / 100.0  # Negative reward proportional to distance
        
        if done:
            # Add completion bonus based on total path efficiency
            normalized_total = self.total_distance / (100.0 * self.num_cities)
            reward += 2.0 * (1.0 - normalized_total)  # Bonus for shorter total paths

        return self.get_normalized_state(), reward, done

    def get_total_distance(self):
        return self.total_distance


class StabilizedActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights
        for layer in self.actor.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state):
        return self.actor(state)


class StabilizedCriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size*2),
            nn.LayerNorm(hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        for layer in self.critic.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state):
        return self.critic(state)


class StabilizedActorCriticTSP:
    def __init__(self, num_cities, hidden_size=128):
        self.num_cities = num_cities
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_size = num_cities * 2 + num_cities + 3
        
        self.actor = StabilizedActorNetwork(input_size, hidden_size, num_cities).to(self.device)
        self.critic = StabilizedCriticNetwork(input_size, hidden_size).to(self.device)
        
        # Smaller learning rates for stability
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        
        # Simple learning rate decay
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)

    def select_action(self, state, available_cities, temperature=1.0):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            logits = self.actor(state_tensor)
            
            # Mask unavailable actions
            mask = torch.zeros_like(logits)
            mask[list(available_cities)] = 1
            masked_logits = logits * mask - 1e9 * (1 - mask)
            
            # Apply temperature scaling
            probs = torch.softmax(masked_logits / temperature, dim=-1)
            
            # Add small exploration noise
            if temperature > 0.5:  # Only add noise during training
                noise = torch.randn_like(probs) * 0.05 * temperature
                probs = torch.softmax(masked_logits / temperature + noise, dim=-1)
            
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum()
            
            action = Categorical(probs).sample()
            return action.item()

    def calculate_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, states, actions, rewards):
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns = self.calculate_returns(rewards)
        
        # Critic update
        value_pred = self.critic(states_tensor).squeeze()
        critic_loss = nn.MSELoss()(value_pred, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Actor update
        advantages = returns - value_pred.detach()
        
        logits = self.actor(states_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions_tensor)
        
        actor_loss = -(log_probs * advantages).mean()
        
        # Add entropy regularization
        entropy = dist.entropy().mean()
        actor_loss -= 0.01 * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()


class MambaActorCriticNetwork(nn.Module):
    def __init__(self, input_size, model_size, hidden_size, layers, output_size):
        super().__init__()

        # Shared Mamba Backbone as Encoder
        self.input_proj = nn.Linear(input_size, model_size)
        self.mamba = nn.Sequential(
            *[Mamba(model_size, hidden_size, 4, 2) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(model_size)

        # Actor Head (Policy) - outputs one logit per city
        self.actor_head = nn.Linear(model_size, 1)

        # Critic Head (Value) - Aggregates and outputs one value for the whole state
        self.critic_head = nn.Sequential(
            nn.Linear(model_size, model_size // 2),
            nn.ReLU(),
            nn.Linear(model_size // 2, 1)
        )

    def forward(self, state):
        # State encoding through Mamba Backbone state: (B, N, I)
        x = self.input_proj(state)
        x = self.mamba(x)
        x = self.norm(x)
        # Actor Head (Policy)
        action_logits = self.actor_head(x).squeeze(-1) # Shape: (B, N)
        # Critic Head (Value)
        aggregated_state = x.mean(dim=1) # Shape: (B, model_size)
        state_value = self.critic_head(aggregated_state).squeeze(-1) # Shape: (B,)
        return action_logits, state_value


class StabilizedMambaTSP:
    def __init__(self, num_cities, model_size, hidden_size, layers):
        self.num_cities = num_cities
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_size = 4 # [x, y, visited, current]
        
        self.model = MambaActorCriticNetwork(
            input_size=input_size,
            model_size=model_size,
            hidden_size=hidden_size,
            layers=layers,
            output_size=num_cities
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def _prepare_state_sequence(self, flat_states, current_cities):
        """
        Helper function to convert the env's flat state vector into 
        a sequence tensor for Mamba.
        
        - flat_states: (batch, 2*n + n + 3) tensor
        - current_cities: (batch) tensor of indices
        """
        n = self.num_cities
        batch_size = flat_states.shape[0]
        
        # Extract coordinates: (batch, 2*n) -> (batch, n, 2)
        coords = flat_states[:, :2*n].reshape(batch_size, n, 2)
        
        # Extract visited vector: (batch, n) -> (batch, n, 1)
        visited = flat_states[:, 2*n : 2*n + n].reshape(batch_size, n, 1)
        
        # Create current city flag: (batch, n, 1)
        current_flag = torch.zeros(batch_size, n, 1, device=self.device)
        # Expand current_cities for scatter_
        current_cities_expanded = current_cities.unsqueeze(-1).unsqueeze(-1)
        # Place a 1.0 at the current city index
        current_flag.scatter_(1, current_cities_expanded, 1.0)
        
        # Concatenate to form the (batch, n, 4) sequence
        return torch.cat([coords, visited, current_flag], dim=2)

    def select_action(self, state, curr_city, available_cities, temperature=1.0):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():

            # Create a batch of 1
            flat_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            current_city_tensor = torch.LongTensor([curr_city]).to(self.device)
            
            # Convert to sequence
            sequence_tensor = self._prepare_state_sequence(flat_state_tensor, current_city_tensor)

            # --- MODEL CALL ---
            # Get logits, ignore value
            logits, _ = self.model(sequence_tensor)
            logits = logits.squeeze(0) # Remove batch dimension

            # Mask unavailable actions
            mask = torch.zeros_like(logits)
            mask[list(available_cities)] = 1
            masked_logits = logits * mask - 1e9 * (1 - mask)
            
            # Apply temperature scaling
            probs = torch.softmax(masked_logits / temperature, dim=-1)
            
            # Add small exploration noise
            if temperature > 0.5:  # Only add noise during training
                noise = torch.randn_like(probs) * 0.05 * temperature
                probs = torch.softmax(masked_logits / temperature + noise, dim=-1)
            
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum()
            
            action = Categorical(probs).sample()
            return action.item()

    def calculate_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, states, current_cities, actions, rewards):
        self.model.train() # Set model to training mode
        
        # --- BATCH PREPARATION ---
        states_tensor = torch.FloatTensor(states).to(self.device)
        current_cities_tensor = torch.LongTensor(current_cities).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)

        # 1. Calculate returns (same as before)
        returns = self.calculate_returns(rewards)
        
        # 2. Convert all flat states in the batch to sequences
        sequence_batch = self._prepare_state_sequence(states_tensor, current_cities_tensor)

        # --- SINGLE FORWARD PASS ---
        # 3. Get both logits and values from ONE model pass
        logits_batch, value_pred = self.model(sequence_batch)

        # 4. Calculate Advantage
        advantages = returns - value_pred.detach() # .detach() is important
        
        # --- LOSS CALCULATION ---
        
        # Critic Loss (How good was the value prediction?)
        critic_loss = nn.MSELoss()(value_pred, returns)

        # Actor Loss (How good was the policy?)
        dist = Categorical(logits=logits_batch) # Use logits directly
        log_probs = dist.log_prob(actions_tensor)
        actor_loss = -(log_probs * advantages).mean()
        
        # Add entropy regularization
        entropy = dist.entropy().mean()
        
        # 5. Combine losses
        # Use standard weights (0.5 for critic, 0.01 for entropy)
        total_loss = actor_loss + (0.5 * critic_loss) - (0.01 * entropy)
        
        # --- BACKPROPAGATION ---
        # 6. Update on the single combined loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def evaluate(env, agent, num_episodes=100):
    total_distances = []
    total_times = []

    for episode in range(num_episodes):
        start_time = time.time()
        state = env.reset()
        current_city = env.current_city
        done = False

        while not done:
            action = agent.select_action(state, current_city, env.available_cities, temperature=0.1)
            next_state, reward, done = env.step(action)
            state = next_state
            current_city = env.current_city

        total_distances.append(env.get_total_distance())
        total_times.append(time.time() - start_time)

    avg_distance = np.mean(total_distances)
    best_distance = np.min(total_distances)
    avg_time = np.mean(total_times)

    print(f"Evaluation Results:\n"
          f"Avg Distance: {avg_distance:.2f}\n"
          f"Best Distance: {best_distance:.2f}\n"
          f"Avg Time per Episode: {avg_time:.2f} sec")

    return avg_distance, best_distance, avg_time

def train(env, agent, num_episodes=10000, eval_interval=500):
    best_distance = float('inf')
    best_episode = 0
    temperature = 1.0  # Start with lower temperature
    
    for episode in range(num_episodes):
        state = env.reset()
        current_city = env.current_city # Get initial city
        done = False

        states, current_cities, actions, rewards = [], [], [], []

        while not done:
            action = agent.select_action(state, current_city, env.available_cities, temperature)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            current_cities.append(current_city)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            current_city = env.current_city 

        # Update policy
        actor_loss, critic_loss = agent.update(states, current_cities, actions, rewards)

        # Update temperature using a more stable schedule
        temperature = max(0.5, 1.0 * np.exp(-episode / 2000))
        
        # Track best performance
        total_distance = env.get_total_distance()
        if total_distance < best_distance:
            best_distance = total_distance
            best_episode = episode

            # Save best model
            torch.save({
                'model_state_dict': agent.model.state_dict(),
                'best_distance': best_distance,
            }, 'best_tsp_model.pth')

        # Step the learning rate scheduler
        agent.scheduler.step()
        
        if episode % eval_interval == 0:
            print(f"Episode {episode}, Actor Loss: {actor_loss:.4f}, "
                  f"Critic Loss: {critic_loss:.4f}, Best Distance: {best_distance:.2f}, "
                  f"Temperature: {temperature:.2f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize environment and agent
    env = TSPEnvironment(num_cities=20)
    agent = StabilizedMambaTSP(num_cities=20, model_size=128, hidden_size=256, layers=2)
    
    # Training
    print("Starting training...")
    train(env, agent)
    
    # Final evaluation
    print("\n===== Final Actor-Critic Evaluation =====")
    ac_avg, ac_best, ac_time = evaluate(env, agent)
    
    # Random baseline comparison
    print("\n===== Random Baseline =====")
    random_agent = StabilizedMambaTSP(num_cities=20, model_size=128, hidden_size=256, layers=2)
    rand_avg, rand_best, rand_time = evaluate(env, random_agent)
    
    # Print final comparison
    print("\n===== Final Results =====")
    print(f"Actor-Critic: Best={ac_best:.2f}, Avg={ac_avg:.2f}, Time={ac_time:.2f}s")
    print(f"Random Baseline: Best={rand_best:.2f}, Avg={rand_avg:.2f}, Time={rand_time:.2f}s")
