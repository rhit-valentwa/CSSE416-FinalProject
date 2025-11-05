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
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
N_EPISODES = 10000
REWARD_HISTORY_SIZE = 100
CHECKPOINT_FREQ = 500
LOG_REWARD_DIR = "logs/rew"
CHECKPOINT_DIR = "checkpoints"
ACTION_SIZE = 8
UPDATE_EPOCHS = 4
ROLLOUT_LENGTH = 2048

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
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)

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

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]
        
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    
    return advantages

# =============================
# PPOAgent Class
# =============================
class PPOAgent:
    
    def __init__(self, action_size, device):
        self.device = device
        self.action_size = action_size
        self.network = ActorCritic(action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.rollout_buffer = RolloutBuffer()
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.clip_epsilon = CLIP_EPSILON
        self.value_coef = VALUE_COEF
        self.entropy_coef = ENTROPY_COEF
        self.max_grad_norm = MAX_GRAD_NORM
        self.update_epochs = UPDATE_EPOCHS

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state.unsqueeze(0))
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, value, log_prob, done):
        self.rollout_buffer.push(
            state.cpu().numpy(),
            action,
            reward,
            value,
            log_prob,
            done
        )

    def train_step(self):
        if len(self.rollout_buffer) == 0:
            return None, None, None
        
        # Prepare batch
        states = torch.FloatTensor(np.array(self.rollout_buffer.states)).to(self.device)
        actions = torch.LongTensor(self.rollout_buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.rollout_buffer.log_probs).to(self.device)
        values = self.rollout_buffer.values
        rewards = self.rollout_buffer.rewards
        dones = self.rollout_buffer.dones
        
        # Compute advantages
        advantages = compute_gae(rewards, values, dones, self.gamma, self.gae_lambda)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.update_epochs):
            # Get current policy and value
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)
            new_values = new_values.squeeze()
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values, returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        return (total_policy_loss / self.update_epochs, 
                total_value_loss / self.update_epochs, 
                total_entropy / self.update_epochs)

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
agent = PPOAgent(ACTION_SIZE, DEVICE)
# agent.load('checkpoints/ppo_episode_500.pth')  # Uncomment to load checkpoint

reward_history = deque(maxlen=REWARD_HISTORY_SIZE)

for episode in range(N_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(DEVICE)
    total_reward = 0
    steps_in_episode = 0
    
    while True:
        steps_in_episode += 1
        
        # Select action using policy
        action_idx, log_prob, value = agent.select_action(state)
        action = index_to_multibinary(action_idx)
        
        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        
        # Store transition
        agent.store_transition(state, action_idx, reward, value, log_prob, done)
        
        total_reward += reward
        state = next_state
        
        # Update policy when buffer is full or episode ends
        if len(agent.rollout_buffer) >= ROLLOUT_LENGTH or done:
            policy_loss, value_loss, entropy = agent.train_step()
            if policy_loss is not None:
                print(f"  Update - Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        
        if done:
            break
    
    reward_history.append(total_reward)
    
    # Print episode info
    print(f"Episode {episode}: Reward={total_reward:.2f}, Steps={steps_in_episode}")
    import os
    os.makedirs(LOG_REWARD_DIR, exist_ok=True)
    log_path_2 = os.path.join(LOG_REWARD_DIR, "episodic_info.txt")
    with open(log_path_2, "a") as f:
        f.write(f"Episode {episode}; Steps: {steps_in_episode}; Ended: {info['death_by']}\n")

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