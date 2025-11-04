import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import os
from gymnasium_env.envs.mario_world import MarioLevelEnv

# =============================
# HYPERPARAMETERS
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Environment
NUMBER_OF_SEQUENTIAL_FRAMES = 6
ACTION_SIZE = 8  # 2^3 combinations of [RIGHT, JUMP, LEFT]

# PPO Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
# ENTROPY_COEF = 0.01
ENTROPY_COEF_START = 0.1  # Much higher starting entropy
ENTROPY_COEF_END = 0.01
ENTROPY_DECAY = 0.9995  # Gradual decay
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_EPISODES = 10000
N_STEPS = 512  # Steps per policy update
N_EPOCHS = 4  # Epochs per update
BATCH_SIZE = 64
MAX_EPISODE_STEPS = 2500
REWARD_HISTORY_SIZE = 100
CHECKPOINT_FREQ = 200
LOG_REWARD_DIR = "logs/ppo_rew"
CHECKPOINT_DIR = "checkpoints/ppo"


# =============================
# ACTOR-CRITIC NETWORK
# =============================
class ActorCritic(nn.Module):
    """
    Shared convolutional backbone with separate actor and critic heads.
    3 convolutional layers as specified.
    """
    def __init__(self, n_actions):
        super().__init__()
        
        # Shared convolutional layers (3 layers)
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
            nn.Linear(512, n_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value
    
    def get_action(self, x, deterministic=False):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(self, x, actions):
        logits, values = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy, values


# =============================
# ROLLOUT BUFFER
# =============================
class RolloutBuffer:
    """Stores trajectories for PPO updates."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.FloatTensor(self.rewards),
            torch.stack(self.values).view(-1),
            torch.FloatTensor(self.dones)
        )


# =============================
# PPO AGENT
# =============================
class PPOAgent:
    def __init__(self, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        
        self.policy = ActorCritic(n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
        
        self.buffer = RolloutBuffer()
        self.entropy_coef = ENTROPY_COEF_START
        self.update_count = 0
        
    def select_action(self, state):
        """Select action using current policy."""
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action(state.unsqueeze(0))
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_state):
        """Update policy using PPO."""
        # Get data from buffer
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        # Compute next value for GAE
        with torch.no_grad():
            _, next_value = self.policy(next_state.unsqueeze(0))
            next_value = next_value.squeeze()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        
        for epoch in range(N_EPOCHS):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, curr_values = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                curr_values = curr_values.squeeze()
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                # value_loss = F.mse_loss(curr_values, batch_returns)

                # Value loss with clipping
                value_pred_clipped = values[batch_indices] + torch.clamp(
                    curr_values - values[batch_indices], -CLIP_EPSILON, CLIP_EPSILON
                )
                value_losses_unclipped = (curr_values - batch_returns) ** 2
                value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_losses_unclipped, value_losses_clipped).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + VALUE_COEF * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                # Approximate KL for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    approx_kls.append(approx_kl.item())
            
            # Early stopping if KL divergence is too high
            if np.mean(approx_kls) > 0.02:
                print(f"  Early stopping at epoch {epoch} due to high KL: {np.mean(approx_kls):.4f}")
                break

        # Decay entropy coefficient
        self.entropy_coef = max(ENTROPY_COEF_END, self.entropy_coef * ENTROPY_DECAY)
        self.update_count += 1

        # Clear buffer
        self.buffer.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'approx_kl': np.mean(approx_kls),
            'entropy_coef': self.entropy_coef
        }
    
    def save(self, episode):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'ppo_episode_{episode}.pth')
        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef,
            'update_count': self.update_count,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
        self.update_count = checkpoint.get('update_count', 0)
        print(f"Loaded checkpoint: {checkpoint_path}")


# =============================
# UTILITY FUNCTIONS
# =============================
def index_to_multibinary(index):
    """Convert action index 0-7 to MultiBinary [right, jump, left]."""
    return np.array([
        (index >> 2) & 1,
        (index >> 1) & 1,
        index & 1
    ])


# =============================
# TRAINING LOOP
# =============================
def train():
    env = MarioLevelEnv(
        render_mode="rgb_array",
        number_of_sequential_frames=NUMBER_OF_SEQUENTIAL_FRAMES,
        max_steps=MAX_EPISODE_STEPS,
    )
    agent = PPOAgent(ACTION_SIZE, DEVICE)
    
    # Optional: Load checkpoint
    agent.load('checkpoints/ppo/ppo_episode_200.pth')
    loaded_episode = 200
    
    reward_history = deque(maxlen=REWARD_HISTORY_SIZE)
    global_step = 0
    
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(DEVICE)
        
        episode_reward = 0
        episode_steps = 0
        steps_since_update = 0
        
        done = False
        while not done:
            # Select action
            action_idx, log_prob, value = agent.select_action(state)
            action = index_to_multibinary(action_idx)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state).to(DEVICE)
            
            # Store transition
            agent.buffer.add(
                state,
                torch.tensor(action_idx),
                torch.tensor(log_prob),
                reward,
                torch.tensor(value),
                done
            )
            
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            steps_since_update += 1
            state = next_state
            
            # Update every N_STEPS or at episode end
            should_update = (steps_since_update >= N_STEPS) or done
            if should_update and len(agent.buffer.states) >= BATCH_SIZE:
                losses = agent.update(state)
                print(f"  Update - Policy: {losses['policy_loss']:.4f}, "
                      f"Value: {losses['value_loss']:.4f}, "
                      f"Entropy: {losses['entropy_loss']:.4f}, "
                      f"KL: {losses['approx_kl']:.4f}, "
                      f"Ent_Coef: {losses['entropy_coef']:.4f}")
                
                # Clear CUDA cache after each update
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                steps_since_update = 0
        
        reward_history.append(episode_reward)
        
        # Memory cleanup after every episode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        print(f"Episode {episode + loaded_episode}: Reward={episode_reward:.2f}, Steps={episode_steps}, Max X={info['x']}\n")
        
        if len(reward_history) == REWARD_HISTORY_SIZE:
            avg_reward = sum(reward_history) / REWARD_HISTORY_SIZE
            print(f"Average Reward (last {REWARD_HISTORY_SIZE}): {avg_reward:.2f}")
            
            os.makedirs(LOG_REWARD_DIR, exist_ok=True)
            log_path = os.path.join(LOG_REWARD_DIR, "avg_reward.txt")
            with open(log_path, "a") as f:
                f.write(f"Episode {episode + loaded_episode}; Average Reward: {avg_reward:.2f}\n")
        
        # Save checkpoint
        if episode % CHECKPOINT_FREQ == 0 and episode > 0:
            agent.save(episode + loaded_episode)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    env.close()


if __name__ == "__main__":
    train()