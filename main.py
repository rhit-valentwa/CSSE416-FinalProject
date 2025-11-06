import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from gymnasium_env.envs.mario_world import MarioLevelEnv

# =============================
# CONSTANTS & HYPERPARAMETERS
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True
NUMBER_OF_SEQUENTIAL_FRAMES = 6
BATCH_SIZE = 256  # Increased for more stable updates
LEARNING_RATE = 2.5e-4  # Standard PPO learning rate
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01  # Consider increasing to 0.02 for more exploration
MAX_GRAD_NORM = 0.5
N_EPISODES = 10000
REWARD_HISTORY_SIZE = 100
CHECKPOINT_FREQ = 500
LOG_REWARD_DIR = "logs/rew"
CHECKPOINT_DIR = "checkpoints"
ACTION_SIZE = 8
UPDATE_EPOCHS = 4
ROLLOUT_LENGTH = 32768
MINIBATCH_SIZE = 256  # For minibatch updates

if PRINT_DEVICE:
    print(f"Using device: {DEVICE}")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        # Shared convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(NUMBER_OF_SEQUENTIAL_FRAMES, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate flattened size
        conv_out_size = self._get_conv_out((NUMBER_OF_SEQUENTIAL_FRAMES, 60, 80))
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        features = self.conv(x)
        features = features.view(x.size(0), -1)
        return self.actor(features), self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value

# Rollout buffer for PPO
class RolloutBuffer:
    def __init__(self, size, state_shape, device):
        self.size = size
        self.device = device
        self.states = torch.zeros((size, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros(size, dtype=torch.long)
        self.rewards = torch.zeros(size, dtype=torch.float32)
        self.values = torch.zeros(size, dtype=torch.float32)
        self.log_probs = torch.zeros(size, dtype=torch.float32)
        self.dones = torch.zeros(size, dtype=torch.float32)
        self.advantages = torch.zeros(size, dtype=torch.float32)
        self.returns = torch.zeros(size, dtype=torch.float32)
        self.ptr = 0
        self.path_start_idx = 0
    
    def push(self, state, action, reward, value, log_prob, done):
        idx = self.ptr % self.size
        self.states[idx] = torch.from_numpy(state) if isinstance(state, np.ndarray) else state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """Compute advantages and returns for the trajectory that just ended"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # Append last_value for bootstrapping
        values_with_bootstrap = torch.cat([values, torch.tensor([last_value])])
        
        # Compute GAE
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values_with_bootstrap[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """Get all data from the buffer"""
        assert self.ptr == self.size, "Buffer not full"
        self.ptr = 0
        self.path_start_idx = 0
        
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        return (
            self.states.to(self.device),
            self.actions.to(self.device),
            self.log_probs.to(self.device),
            self.advantages.to(self.device),
            self.returns.to(self.device),
        )
    
    def is_full(self):
        return self.ptr >= self.size

def multibinary_to_index(action):
    """Convert [a, b, c] to single index 0-7"""
    return action[0] * 4 + action[1] * 2 + action[2]

def index_to_multibinary(index):
    """Convert index 0-7 to [a, b, c]"""
    return np.array([
        (index >> 2) & 1,
        (index >> 1) & 1,
        index & 1
    ])

# =============================
# PPOAgent Class
# =============================
class PPOAgent:
    
    def __init__(self, action_size, device, state_shape):
        self.device = device
        self.action_size = action_size
        self.network = ActorCritic(action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.rollout_buffer = RolloutBuffer(ROLLOUT_LENGTH, state_shape, device)
        self.clip_epsilon = CLIP_EPSILON
        self.value_coef = VALUE_COEF
        self.entropy_coef = ENTROPY_COEF
        self.max_grad_norm = MAX_GRAD_NORM
        self.update_epochs = UPDATE_EPOCHS

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = state.unsqueeze(0) if state.dim() == 3 else state
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, value, log_prob, done):
        state_np = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        self.rollout_buffer.push(state_np, action, reward, value, log_prob, done)

    def train_step(self):
        # Get data from buffer
        states, actions, old_log_probs, advantages, returns = self.rollout_buffer.get()
        
        # PPO update with minibatches
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        n_updates = 0
        
        for epoch in range(self.update_epochs):
            # Create random minibatches
            indices = torch.randperm(ROLLOUT_LENGTH)
            
            for start in range(0, ROLLOUT_LENGTH, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Get current policy and value
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )
                new_values = new_values.squeeze()
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping (optional but helps stability)
                value_loss = ((new_values - mb_returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Track clipping fraction
                with torch.no_grad():
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()
                    total_clipfrac += clipfrac
                
                n_updates += 1
        
        return (
            total_policy_loss / n_updates,
            total_value_loss / n_updates,
            total_entropy / n_updates,
            total_clipfrac / n_updates
        )

    def save(self, episode):
        import os
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'ppo_episode_{episode}.pth')
        torch.save({
            'episode': episode,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        """Load a saved checkpoint into the agent's network and optimizer."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# =============================
# Training Loop
# =============================
env = MarioLevelEnv(render_mode="human", number_of_sequential_frames=NUMBER_OF_SEQUENTIAL_FRAMES)
state_shape = (NUMBER_OF_SEQUENTIAL_FRAMES, 60, 80)
agent = PPOAgent(ACTION_SIZE, DEVICE, state_shape)

import os
os.makedirs(LOG_REWARD_DIR, exist_ok=True)
log_path_2 = os.path.join(LOG_REWARD_DIR, "episodic_info.txt")
log_path_3 = os.path.join(LOG_REWARD_DIR, "policy_info.txt")
agent.load('checkpoints/ppo_episode_1000.pth')  # Uncomment to load checkpoint

reward_history = deque(maxlen=REWARD_HISTORY_SIZE)
global_step = 0

for episode in range(1001, N_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(DEVICE)
    episode_reward = 0
    episode_steps = 0
    
    while True:
        # Select action using policy
        action_idx, log_prob, value = agent.select_action(state)
        action = index_to_multibinary(action_idx)
        
        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        
        # Store transition
        agent.store_transition(state, action_idx, reward, value, log_prob, done)
        
        episode_reward += reward
        episode_steps += 1
        global_step += 1
        state = next_state
        
        # When episode ends, compute advantages for this trajectory
        if done:
            agent.rollout_buffer.finish_path(last_value=0)
            break
        
        # If buffer is full, train and continue episode
        if agent.rollout_buffer.is_full():
            # Bootstrap with current state's value
            with torch.no_grad():
                _, _, _, last_value = agent.network.get_action_and_value(next_state.unsqueeze(0))
            agent.rollout_buffer.finish_path(last_value=last_value.item())
            
            # Train on collected data
            policy_loss, value_loss, entropy, clipfrac = agent.train_step()
            global_step = 0
            print(f"  Update - Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                  f"Entropy: {entropy:.4f}, ClipFrac: {clipfrac:.3f}")
            with open(log_path_3, "a") as f:
                f.write(f"Update - Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}, ClipFrac: {clipfrac:.3f}\n")
            
            # Continue episode with fresh buffer (don't break!)
    
    reward_history.append(episode_reward)
    # Print episode info
    print(f"\nEpisode {episode}: Reward={episode_reward:.2f}, Steps={episode_steps}, "
          f"Global Steps={global_step}")
    
    with open(log_path_2, "a") as f:
        f.write(f"Episode {episode}; Steps: {episode_steps}; Ended: {info['death_by']}\n")

    if len(reward_history) == REWARD_HISTORY_SIZE:
        avg_reward = sum(reward_history)/REWARD_HISTORY_SIZE
        print(f"Episode {episode}; Average Reward: {avg_reward:.2f}")
        log_path = os.path.join(LOG_REWARD_DIR, "avg_reward.txt")
        with open(log_path, "a") as f:
            f.write(f"Episode {episode}; Average Reward: {avg_reward:.2f}\n")
    
    if episode % 10 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    if episode % CHECKPOINT_FREQ == 0 and episode > 0:
        agent.save(episode)

env.close()