import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from gymnasium_env.envs.mario_world import MarioLevelEnv

# Simple Q-network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4800, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, action_size)
        )
    
    def forward(self, x):
        # x = x.T
        x = (x.flatten()).reshape(1, 4800)
        # print(x.shape)
        return self.fc(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # return self.buffer[-1]
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

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

# Training loop
env = MarioLevelEnv(render_mode="human")
state_size = 60*80 # env.observation_space.shape[0]  # adjust based on your state
action_size = 8 # env.action_space.n

q_network = DQN(state_size, action_size)
target_network = DQN(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=1e-2)
replay_buffer = ReplayBuffer(10000)

gamma = 0.99  # discount factor
epsilon = 0.34  # exploration rate
epsilon_decay = 0.990
epsilon_min = 0.01
batch_size = 1

for episode in range(1000):
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    
    while True:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state)  # shape: [1, 8]
                action_idx = q_values.argmax().item()
                action = index_to_multibinary(action_idx)
                # action = q_network(state) # .argmax().item()
                # print("action:", action)
                # print("argmax:", action.argmax())
                # print("item:", action.argmax().item())
        
        # Take action
        # print(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state)
        
        # Store in replay buffer
        action_idx = multibinary_to_index(action)
        replay_buffer.push(state, action_idx, reward, next_state, done)
        
        # Train if enough samples
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            states = torch.stack([s for s, _, _, _, _ in batch])
            actions = torch.LongTensor([a for _, a, _, _, _ in batch])
            rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
            next_states = torch.stack([s for _, _, _, s, _ in batch])
            dones = torch.FloatTensor([d for _, _, _, _, d in batch])
            
            # Current Q values
            # print(states.shape)
            # print(actions.unsqueeze(1))
            # print(q_network(states))
            current_q = q_network(states)#.gather(3, actions.unsqueeze(1))
            
            # Target Q values
            with torch.no_grad():
                max_next_q = target_network(next_states).max(1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)
            
            # Compute loss and update
            loss = nn.MSELoss()(current_q, target_q) # possibly need to change this part?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Update target network periodically
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")