import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from gymnasium_env.envs.mario_world import MarioLevelEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Deep Q-Network
# This is the very same network structure used in the DQN paper
# https://arxiv.org/abs/1312.5602
class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=8, stride=4),
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
        conv_out_size = self._get_conv_out((6, 60, 80))
        
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
        #return list(self.buffer)[-batch_size:]
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

# Training loop
env = MarioLevelEnv(render_mode="human")
# state_size = 75*100 # env.observation_space.shape[0]
action_size = 8 # env.action_space.n



q_network = DQN(action_size).to(device)
target_network = DQN(action_size).to(device)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=5e-6)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
replay_buffer = ReplayBuffer(50000)

# checkpoint = torch.load('oct_26_night_episode_500.pth')
# q_network.load_state_dict(checkpoint['q_network_state_dict'])
# target_network.load_state_dict(checkpoint['target_network_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

gamma = 0.99  # discount factor
epsilon = 0.5  # exploration rate
epsilon_decay = 0.5
epsilon_min = 0.1
batch_size = 32

N = 100
reward_history = deque(maxlen=N)
step = 0
for episode in range(10000):
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(device)  # Add batch and channel dimensions
    # print("State initial:",state.shape)
    total_reward = 0
    while True:
        step += 1
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state.unsqueeze(0))  # shape: [1, 8]
                action_idx = q_values.argmax().item()
                action = index_to_multibinary(action_idx)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).to(device)
        # if( step % 10 == 0):
        #     print(f"Reward: {reward}")
        # print("Before push:", state.shape)
        # Store in replay buffer
        replay_buffer.push(
            state.cpu().numpy(),
            multibinary_to_index(action), 
            reward, 
            next_state.cpu().numpy(),
            done
        )
        
        # Train if enough samples
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            states = torch.FloatTensor(np.array([s for s, _, _, _, _ in batch])).to(device)
            actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(device)
            rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(device)
            next_states = torch.FloatTensor(np.array([s for _, _, _, s, _ in batch])).to(device)
            dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(device)
                        
            # print("States batch shape:", states.shape)
            current_q_all = q_network(states)  # [batch_size, 8]
            current_q = current_q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q values
            with torch.no_grad():
                max_next_q = target_network(next_states).max(1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)
                # avg_q = q_network(states)
                # max_q = q_network(states).max().item()

            # if(episode > 25):
            #     print(f"Q: {avg_q}, Max Q: {max_q:.2f}, target: {target_q}")

            # Compute loss and update
            loss = nn.MSELoss()(current_q, target_q) # possibly need to change this part?
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler.step()
        
        total_reward += reward
        state = next_state # torch.FloatTensor(next_state.cpu().numpy()).unsqueeze(0).to(device)
        
        if done:
            break
    
    # Decay exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)
    if( len(reward_history) == N):
        avg_reward = sum(reward_history)/N
        print(f"Episode {episode}; Average Reward: {avg_reward}")
    # Update target network periodically
    if episode % 10 == 0 and step > 5000:
        # target_network.load_state_dict(q_network.state_dict())
        step = 0
        tau = 0.005  # Small value = slower updates
        for target_param, param in zip(target_network.parameters(), q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    if episode % 2 == 0:
        torch.cuda.empty_cache()  # Clear CUDA cache
        import gc
        gc.collect()  # Force garbage collection

    if episode % 500 == 0:  # Save every 500 episodes
        torch.save({
            'episode': episode,
            'q_network_state_dict': q_network.state_dict(),
            'target_network_state_dict': target_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
        }, f'oct_26_night_episode_{episode}.pth')
    
    # print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")