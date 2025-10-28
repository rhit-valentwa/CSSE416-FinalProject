import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from gymnasium_env.envs.mario_world import MarioLevelEnv

# =============================
# CONSTANTS & HYPERPARAMETERS
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True
NUMBER_OF_SEQUENTIAL_FRAMES = 6
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 32
LEARNING_RATE = 5e-6
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_DECAY = 0.5
EPSILON_MIN = 0.1
TAU = 0.005
N_EPISODES = 10000
TARGET_UPDATE_FREQ = 10
REWARD_HISTORY_SIZE = 100
CHECKPOINT_FREQ = 500
LOG_REWARD_DIR = "logs/rew"
CHECKPOINT_DIR = "checkpoints"
ACTION_SIZE = 8

if PRINT_DEVICE:
    print(f"Using device: {DEVICE}")

# Deep Q-Network
# This is the very same network structure used in the DQN paper
# https://arxiv.org/abs/1312.5602
class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUMBER_OF_SEQUENTIAL_FRAMES, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Automatically calculate flattened size
        conv_out_size = self._get_conv_out((NUMBER_OF_SEQUENTIAL_FRAMES, 60, 80))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size(0), -1)  # Flatten
        return self.fc(conv_out)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=1000):
        capacity = capacity//2
        self.buffer_neg = deque(maxlen=capacity)
        self.buffer_pos = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        if reward < 0:
            self.buffer_neg.append((state, action, reward, next_state, done))
        else:
            self.buffer_pos.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer_neg, batch_size//2) + random.sample(self.buffer_pos, batch_size//2)
    def __len__(self):
        return min(len(self.buffer_neg), len(self.buffer_pos))

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
# DQNAgent Class
# =============================
class DQNAgent:
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
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, episode):
        import os
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'oct_27_night_episode_{episode}.pth')
        torch.save({
            'episode': episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, checkpoint_path)

# =============================
# Training Loop
# =============================
env = MarioLevelEnv(render_mode="human", number_of_sequential_frames=NUMBER_OF_SEQUENTIAL_FRAMES)
agent = DQNAgent(ACTION_SIZE, DEVICE)

reward_history = deque(maxlen=REWARD_HISTORY_SIZE)
step = 0
for episode in range(N_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(DEVICE)
    total_reward = 0
    while True:
        step += 1
        action = agent.select_action(state, env)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        agent.update_target_network()  # Soft update every step
        total_reward += reward
        state = next_state
        if done:
            break
    print(f"Episode {episode} finished with reward {total_reward}")
    agent.decay_epsilon()
    reward_history.append(total_reward)
    if len(reward_history) == REWARD_HISTORY_SIZE:
        avg_reward = sum(reward_history)/REWARD_HISTORY_SIZE
        print(f"Episode {episode}; Average Reward: {avg_reward}")
        # Save average reward to logs/rew directory
        import os
        os.makedirs(LOG_REWARD_DIR, exist_ok=True)
        log_path = os.path.join(LOG_REWARD_DIR, "avg_reward.txt")
        with open(log_path, "a") as f:
            f.write(f"Episode {episode}; Average Reward: {avg_reward}\n")
    if episode % 2 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    if episode % CHECKPOINT_FREQ == 0:
        agent.save(episode)