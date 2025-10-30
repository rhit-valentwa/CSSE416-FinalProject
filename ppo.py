import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import os
from gymnasium_env.envs.mario_world import MarioLevelEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

importing = False

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        
        # Simplified network for (4, 60, 80) input
        # Input: (4, 60, 80)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),  # -> (32, 14, 19)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # -> (64, 6, 8)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # -> (64, 4, 6)
            nn.ReLU(),
            nn.Flatten(),  # -> 1536
        )
        
        # Actor head (policy) - outputs logits for 8 actions
        self.actor = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        
        # Critic head (value function) - outputs state value
        self.critic = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
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
    def __init__(self, action_size, lr=3e-4, gamma=0.99, eps_clip=0.1, 
                 k_epochs=3, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01):
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


def evaluate_model(agent, env, n_episodes=5):
    """Evaluate agent and return rewards"""
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        episode_reward = 0
        
        while True:
            action_idx, _, _ = agent.select_action(state)
            action = index_to_multibinary(action_idx)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = torch.FloatTensor(next_state).to(device)
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
    
    return rewards if n_episodes > 1 else rewards[0]


# Training loop
def train_ppo():
    # Better reward shaping to prevent "run right and die" strategy
    reward_config = {
        "dx_scale": 0.1,        # Small movement reward (prevents reward hacking)
        "score_scale": 0.01,     # Reward actual game score
        "death_penalty": -150.0,   # Strong death penalty
        "win_bonus": 1000.0,       # Large win bonus
        "progress_milestone": 10.0,  # Reward for reaching progress milestones
    }
    
    # Create environment with your custom settings
    env = MarioLevelEnv(
        render_mode="human",  # Use "human" to watch, "rgb_array" for faster training
        width=800,
        height=600,
        max_steps=2000,           # Prevent infinite episodes
        frame_skip=4,
        number_of_sequential_frames=4,
        reward_cfg=reward_config,
    )
    
    action_size = 8  # 2^3 combinations of [Right, Jump, Left]
    
    agent = PPOAgent(
        action_size=action_size,
        lr=5e-5,
        gamma=0.99,
        eps_clip=0.1,    # Conservative clipping
        k_epochs=3,      # Moderate updates
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.1,   # Starting entropy coefficient
    )
    
    # Initialize tracking variables
    loaded_episode = 0
    best_reward = -float('inf')
    best_avg_reward = -float('inf')
    episode_rewards = deque(maxlen=100)
    
    if (importing):
        # FIRST: Load the checkpoint you want to continue training from
        checkpoint_file = 'ppo_checkpoint_episode_200.pth'
        if os.path.exists(checkpoint_file):
            agent.load(checkpoint_file)
            loaded_episode = 200
            print(f"✅ Loaded checkpoint from episode {loaded_episode}")
        
        # SECOND: Evaluate best models to get baseline scores (FAST evaluation)
        if os.path.exists('ppo_best_episode.pth'):
            print("📊 Evaluating best episode model...")
            temp_agent = PPOAgent(action_size=action_size)
            temp_agent.load('ppo_best_episode.pth')
            eval_reward = evaluate_model(temp_agent, env, n_episodes=1)
            best_reward = eval_reward
            print(f"  Best episode reward: {best_reward:.2f}")
            del temp_agent  # Free memory
        
        if os.path.exists('ppo_best_avg.pth'):
            print("📊 Evaluating best avg model...")
            temp_agent = PPOAgent(action_size=action_size)
            temp_agent.load('ppo_best_avg.pth')
            eval_rewards = evaluate_model(temp_agent, env, n_episodes=3)  # Reduced from 20 to 3
            best_avg_reward = np.mean(eval_rewards)
            print(f"  Best avg reward: {best_avg_reward:.2f}")
            del temp_agent  # Free memory
    
    print("\n🚀 Starting training...\n")
    
    # Hyperparameters
    update_frequency = 512      # Update every 512 steps
    max_episodes = 10000
    min_buffer_size = 128       # Minimum buffer size for episode-end updates

    # Entropy decay schedule
    ent_coef_start = agent.ent_coef  # High initial exploration
    ent_coef_end = 0.02         # Final exploration level
    ent_coef_decay = 0.99995      # Slow decay
    current_ent_coef = ent_coef_start

    buffer = RolloutBuffer()
    timestep = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        
        episode_reward = 0
        episode_steps = 0

        # Update entropy coefficient for this episode
        agent.ent_coef = max(ent_coef_end, current_ent_coef)
        current_ent_coef *= ent_coef_decay
        
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
            
            # Update policy regularly (like DQN does every step)
            if len(buffer) >= update_frequency:
                losses = agent.update(buffer, state)
                buffer.clear()
                
                print(f"  [Update@{timestep}] Actor: {losses['actor_loss']:.4f}, "
                      f"Critic: {losses['critic_loss']:.4f}, "
                      f"Entropy: {losses['entropy']:.4f} (coef: {agent.ent_coef:.4f})")
                
                # Warn if entropy is too low
                if losses['entropy'] < 0.5:
                    print(f"  ⚠️  Low entropy detected! Policy may be too deterministic.")
            
            if done:
                # Always update at episode end if we have enough samples
                if len(buffer) >= min_buffer_size:
                    buffer_size = len(buffer)
                    losses = agent.update(buffer, state)
                    buffer.clear()
                    # print(f"  [EpisodeEnd] Updated with {buffer_size} samples")
                break
        
        # Episode summary
        # episode_rewards.append(episode_reward)
        # avg_reward_100 = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
        actual_episode = episode + loaded_episode
        
        # Calculate moving average and trend
        # if len(episode_rewards) >= 20:
        #     recent_20 = np.mean(list(episode_rewards)[-20:])
        #     older_20 = np.mean(list(episode_rewards)[-40:-20]) if len(episode_rewards) >= 40 else recent_20
        #     trend = "📈" if recent_20 > older_20 else "📉" if recent_20 < older_20 else "➡️"
        # else:
        #     trend = "➡️"

        print(f"Ep {actual_episode:4d} | Steps: {episode_steps:4d} | "
              f"Reward: {episode_reward:7.2f} | Score: {info['score']:4d} | "
              f"X: {info['x']:4d} | Ent: {agent.ent_coef:.4f}")

        # Detect if stuck (same low reward repeatedly)
        # if len(episode_rewards) >= 5:
        #     recent_std = np.std(list(episode_rewards)[-10:])
        #     recent_mean = np.mean(list(episode_rewards)[-10:])
            
        #     # Check for degenerate "run right and die" policy
        #     if recent_std < 5.0 and recent_mean < 80 and episode_steps < 150:
        #         print(f"  🚨 DETECTED DEGENERATE POLICY! (mean={recent_mean:.2f}, steps={episode_steps})")
                
        #         # Hard reset
        #         agent.policy = ActorCritic(action_size).to(device)
        #         agent.policy_old = ActorCritic(action_size).to(device)
        #         agent.policy_old.load_state_dict(agent.policy.state_dict())
        #         agent.optimizer = optim.Adam(agent.policy.parameters(), lr=3e-4)
        #         current_ent_coef = ent_coef_start
        #         episode_rewards.clear()
        #         buffer.clear()
        #         continue
            
        #     # Original stuck detection (low negative rewards)
        #     if recent_std < 1.0 and recent_mean < -5.0:
        #         # print(f"  ⚠️  WARNING: Agent might be stuck! (std={recent_std:.2f}, mean={recent_mean:.2f})")
                
        #         # Auto-recovery: reinitialize policy if stuck for too long
        #         if len(episode_rewards) >= 50 and np.mean(list(episode_rewards)[-50:]) < -5.0:
        #             print("  🔄 AUTO-RECOVERY: Reinitializing policy due to persistent poor performance")
        #             agent.policy = ActorCritic(action_size).to(device)
        #             agent.policy_old = ActorCritic(action_size).to(device)
        #             agent.policy_old.load_state_dict(agent.policy.state_dict())
        #             agent.optimizer = optim.Adam(agent.policy.parameters(), lr=3e-4)
        #             current_ent_coef = ent_coef_start
        #             episode_rewards.clear()
        #             buffer.clear()
        
        # Save best single episode
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('ppo_best_episode.pth')
            print(f"  ✓ New best single episode: {best_reward:.2f}")
        
        # Save best average model (more reliable)
        if len(episode_rewards) >= 20:
            avg_20 = np.mean(list(episode_rewards)[-20:])
            if avg_20 > best_avg_reward:
                best_avg_reward = avg_20
                agent.save('ppo_best_avg.pth')
                print(f"  ✓✓ New best avg-20: {best_avg_reward:.2f}")
        
        # Save checkpoint periodically
        if episode % 200 == 0 and episode > 0:
            agent.save(f'ppo_checkpoint_episode_{actual_episode}.pth')
            print(f"  💾 Checkpoint saved at episode {actual_episode}")
    
    env.close()
    return agent


if __name__ == '__main__':
    agent = train_ppo()