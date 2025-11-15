"""
DQN agent for Mario environment using experience replay and target networks.
Based on: https://arxiv.org/abs/1312.5602
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from gymnasium_env.envs.mario_world import MarioLevelEnv
import log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True
NUMBER_OF_SEQUENTIAL_FRAMES = 6
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
TAU = 0.001  # Soft update rate for target network
N_EPISODES = 10000
REWARD_HISTORY_SIZE = 100
CHECKPOINT_FREQ = 500
LOG_REWARD_DIR = "logs/rew"
CHECKPOINT_DIR = "checkpoints"
ACTION_SIZE = 8
MIN_REPLAY_SIZE = 1000  # Wait for diverse experiences before training

if PRINT_DEVICE:
    print(f"Using device: {DEVICE}")


class DQN(nn.Module):
    """Convolutional Q-network approximating Q(s, a) over stacked frames."""

    def __init__(self, action_size):
        super().__init__()
        # Convolutional feature extractor (similar to Atari DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(NUMBER_OF_SEQUENTIAL_FRAMES, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # We assume env returns (60, 80) grayscale frames, stacked along channel dimension
        conv_out_size = self._get_conv_out((NUMBER_OF_SEQUENTIAL_FRAMES, 60, 80))

        # Fully connected layers mapping features -> Q-values for each discrete action
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def _get_conv_out(self, shape):
        """Calculate flattened conv output size for a given (C, H, W) input shape."""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Forward pass: images -> conv features -> flattened -> fully connected -> Q-values.
        """
        conv_out = self.conv(x)                  # (B, C', H', W')
        conv_out = conv_out.view(x.size(0), -1)  # Flatten to (B, N)
        q_values = self.fc(conv_out)             # (B, action_size)
        return q_values

class ReplayBuffer:
    """Circular buffer for experience replay."""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


def multibinary_to_index(action):
    """Convert [right, jump, sprint] to index 0-7."""
    return action[0] * 4 + action[1] * 2 + action[2]


def index_to_multibinary(index):
    """Convert index 0-7 to [right, jump, sprint]."""
    return np.array([
        (index >> 2) & 1,
        (index >> 1) & 1,
        index & 1
    ])


def try_render_rgb(env):
    """Return RGB array if available, else None."""
    try:
        print(env.render())
        return env.render()
    except Exception:
        return None


class DQNAgent:
    """DQN agent with epsilon-greedy exploration and soft target updates."""
    
    def __init__(self, action_size, device):
        self.device = device
        self.action_size = action_size
        self.q_network = DQN(action_size).to(device)
        self.target_network = DQN(action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch_size = BATCH_SIZE
        self.tau = TAU

    def select_action(self, state, env):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            action_idx = q_values.argmax().item()
            return index_to_multibinary(action_idx)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(
            state.cpu().numpy(),
            multibinary_to_index(action),
            reward,
            next_state.cpu().numpy(),
            done
        )

    def train_step(self):
        """Sample batch and perform one gradient descent step."""
        if len(self.replay_buffer) <= self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([s for s, _, _, _, _ in batch])).to(self.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([s for _, _, _, s, _ in batch])).to(self.device)
        dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(self.device)

        current_q_all = self.q_network(states)
        current_q = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target using target network
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        """Soft update: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, episode):
        import os
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'oct_28_night_episode_{episode+1500}.pth')
        torch.save({
            'episode': episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, checkpoint_path)

    def load(self, checkpoint_path):
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)


# Training loop
env = MarioLevelEnv(render_mode="human", number_of_sequential_frames=NUMBER_OF_SEQUENTIAL_FRAMES)
agent = DQNAgent(ACTION_SIZE, DEVICE)
agent.load('checkpoints/solid_saves/better_4000.pth')

reward_history = deque(maxlen=REWARD_HISTORY_SIZE)

for episode in range(N_EPISODES):
    # Reset environment and initialize episode state
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(DEVICE)
    total_reward = 0
    steps_in_episode = 0

    while True:
        steps_in_episode += 1

        # Choose action (epsilon-greedy) and step environment
        action = agent.select_action(state, env)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).to(DEVICE)

        # log Q-values for current state (for logging)
        with torch.no_grad():
            q_vals = agent.q_network(state.unsqueeze(0)).squeeze(0)

        # Extract some extra info from the environment (position, frame and action index)
        x = info.get("x") if isinstance(info, dict) else None
        y = info.get("y") if isinstance(info, dict) else None
        action_idx = int(multibinary_to_index(action))
        frame_arr = env.render_fullframe()

        # Store transition and train agent (currently commented out for evaluation)
        # agent.store_transition(state, action, reward, next_state, done)
        # if len(agent.replay_buffer) >= MIN_REPLAY_SIZE:
        #     agent.train_step()
        #     agent.update_target_network()

        total_reward += reward
        state = next_state

        if done:
            break

    # agent.decay_epsilon()  # Uncomment for training runs to slowly reduce exploration

    reward_history.append(total_reward)

    print(f"Episode {episode}: Reward={total_reward:.2f}, Steps={steps_in_episode}, Epsilon={agent.epsilon:.4f}")

    if len(reward_history) == REWARD_HISTORY_SIZE:
        avg_reward = sum(reward_history)/REWARD_HISTORY_SIZE
        print(f"Episode {episode}; Average Reward: {avg_reward:.2f}")
        
        # import os
        # os.makedirs(LOG_REWARD_DIR, exist_ok=True)
        # log_path = os.path.join(LOG_REWARD_DIR, "avg_reward.txt")
        # with open(log_path, "a") as f:
        #     f.write(f"Episode {episode}; Average Reward: {avg_reward:.2f}\n")
    
    if episode % 10 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # if episode % CHECKPOINT_FREQ == 0 and episode > 0:
    #     agent.save(episode)