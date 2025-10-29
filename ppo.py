import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from gymnasium_env.envs.mario_world import MarioLevelEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        
        # Input: (4, 60, 80) - 4 stacked grayscale frames
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (32, 30, 40)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (64, 15, 20)
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (64, 7, 10)
            nn.Flatten(),  # -> 4480
        )
        
        # Actor head (policy) - outputs logits for 8 actions
        self.actor = nn.Sequential(
            nn.Linear(4480, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, action_size)
        )
        
        # Critic head (value function) - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(4480, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
    
    def get_action(self, x, deterministic=False):
        """Sample action from policy"""
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value, entropy


def index_to_multibinary(index):
    """Convert index 0-7 to [a, b, c] for MultiBinary(3) action space"""
    return np.array([
        (index >> 2) & 1,  # Right (bit 2)
        (index >> 1) & 1,  # Jump (bit 1)
        index & 1          # Left (bit 0)
    ], dtype=np.int8)


# Rollout buffer for storing trajectories
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def get(self):
        return (
            torch.stack(self.states),
            torch.tensor(self.actions, dtype=torch.long),
            torch.stack(self.log_probs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.tensor(self.dones, dtype=torch.float32),
            torch.stack(self.values)
        )
    
    def __len__(self):
        return len(self.states)


# PPO Agent
class PPOAgent:
    def __init__(self, action_size, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        
        self.policy = ActorCritic(action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Old policy for computing ratio
        self.policy_old = ActorCritic(action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval()
    
    def select_action(self, state):
        """Select action using old policy (for rollout collection)"""
        with torch.no_grad():
            action, log_prob, value, _ = self.policy_old.get_action(state.unsqueeze(0))
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values_list = values.tolist() + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_list[t + 1] * (1 - dones[t]) - values_list[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def update(self, buffer, next_state):
        """Update policy using PPO"""
        # Get data from buffer
        states, actions, old_log_probs, rewards, dones, values = buffer.get()
        
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        values = values.squeeze().to(device)
        
        # Compute next value for GAE
        with torch.no_grad():
            _, next_value = self.policy_old(next_state.unsqueeze(0).to(device))
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values, dones, next_value).to(device)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for K epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.k_epochs):
            # Evaluate actions with current policy
            logits, state_values = self.policy(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Compute ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss (clipped for stability)
            value_pred = state_values.squeeze()
            value_pred_clipped = values + torch.clamp(
                value_pred - values, -self.eps_clip, self.eps_clip
            )
            value_loss1 = (value_pred - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
            
            # Entropy bonus (for exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()
        
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return {
            'actor_loss': total_actor_loss / self.k_epochs,
            'critic_loss': total_critic_loss / self.k_epochs,
            'entropy': total_entropy / self.k_epochs
        }
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Training loop
def train_ppo():
    # Create environment with your custom settings
    env = MarioLevelEnv(
        render_mode="human",  # Change to "rgb_array" for faster training
        width=800,
        height=600,
        max_steps=20000,
        frame_skip=4,
        number_of_sequential_frames=4,
    )
    
    action_size = 8  # 2^3 combinations of [Right, Jump, Left]
    
    agent = PPOAgent(
        action_size=action_size,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.05 # higher = more exploration
    )
    
    # Optional: Load checkpoint
    # agent.load('ppo_checkpoint_episode_200.pth')
    
    # Hyperparameters
    update_timestep = 512  # Update policy every n timesteps
    max_episodes = 10000
    min_buffer_size = 128

    # Entropy decay params
    min_entropy = 0.01
    entropy_decay = 0.995

    buffer = RolloutBuffer()
    timestep = 0
    best_reward = -float('inf')
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        # State is already (4, 60, 80)
        state = torch.FloatTensor(state).to(device)
        
        episode_reward = 0
        episode_steps = 0

        # Update entropy
        agent.ent_coef = max(min_entropy, agent.ent_coef * entropy_decay)
        
        while True:
            timestep += 1
            episode_steps += 1
            
            # Select action from policy
            action_idx, log_prob, value = agent.select_action(state)
            action = index_to_multibinary(action_idx)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = torch.FloatTensor(next_state).to(device)
            
            # Store in buffer
            buffer.add(
                state.cpu(),
                action_idx,
                torch.tensor(log_prob),
                reward,
                done,
                torch.tensor(value)
            )
            
            episode_reward += reward
            state = next_state
            
            # Update policy when buffer is full
            if len(buffer) >= update_timestep:
                losses = agent.update(buffer, state)
                buffer.clear()
                
                print(f"  [Update] Actor Loss: {losses['actor_loss']:.4f}, "
                      f"Critic Loss: {losses['critic_loss']:.4f}, "
                      f"Entropy: {losses['entropy']:.4f}")
                
                # Clear cache
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            if done:
                if len(buffer) >= min_buffer_size:
                    losses = agent.update(buffer, state)
                    buffer.clear()
                    print(f"  [EpisodeEnd] Buffer had {len(buffer)} samples, updated")
                break
        
        # Episode summary
        print(f"Episode {episode:4d} | Steps: {episode_steps:4d} | "
              f"Reward: {episode_reward:7.2f} | Score: {info['score']:4d} | "
              f"X: {info['x']:4d} | Timestep: {timestep}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('ppo_best_model.pth')
            print(f"  ✓ New best reward: {best_reward:.2f}")
        
        # Save checkpoint periodically
        if episode % 200 == 0 and episode > 0:
            agent.save(f'ppo_checkpoint_episode_{episode}.pth')
            print(f"  💾 Checkpoint saved at episode {episode}")
    
    env.close()
    return agent


if __name__ == '__main__':
    agent = train_ppo()