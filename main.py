from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
import cv2

# ============================================================
# RENDERING CONTROL
# ============================================================
ENABLE_RENDERING = False  # Set to False to train faster without rendering
RENDER_EVERY_N_EPISODES = 1  # Render every N episodes (set to 1 to render all)
# ============================================================

# Preprocessing wrapper for frames
class FramePreprocessor:
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height
    
    def preprocess(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84 (standard for DQN)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        return normalized

# Frame stacking wrapper
class FrameStack:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, frame):
        preprocessed = frame
        for _ in range(self.num_frames):
            self.frames.append(preprocessed)
        return self.get_state()
    
    def step(self, frame):
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        return np.stack(self.frames, axis=0)

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Deep Q-Network (Fixed architecture)
class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Automatically calculate flattened size (for 84x84 input)
        conv_out_size = self._get_conv_out((4, 84, 84))
        
        # Fixed: Removed bottleneck - go directly from 512 to action_size
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size(0), -1)
        return self.fc(conv_out)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Initialize everything
action_size = env.action_space.n
preprocessor = FramePreprocessor()
frame_stack = FrameStack(num_frames=4)

q_network = DQN(action_size).to(device)
target_network = DQN(action_size).to(device)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=0.00025)
replay_buffer = ReplayBuffer(100000)

# Try to load checkpoint with compatibility check
import os
start_episode = 0
epsilon = 1.0

checkpoint_files = [
    'mario_dqn_latest.pth',
    'atari_dqn_episode_200.pth',
]

checkpoint_loaded = False
# for checkpoint_file in checkpoint_files:
#     if os.path.exists(checkpoint_file):
#         print(f"Found checkpoint: {checkpoint_file}")
#         try:
#             checkpoint = torch.load(checkpoint_file, map_location=device)
            
#             # Try to load the state dict
#             q_network.load_state_dict(checkpoint['q_network_state_dict'])
#             target_network.load_state_dict(checkpoint['target_network_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             start_episode = checkpoint.get('episode', 0)
#             epsilon = checkpoint.get('epsilon', 1.0)
            
#             print(f"✓ Successfully loaded checkpoint from episode {start_episode}")
#             checkpoint_loaded = True
#             break
            
#         except RuntimeError as e:
#             print(f"✗ Checkpoint incompatible with current architecture: {checkpoint_file}")
#             print(f"  Error: {str(e)[:100]}...")
#             print(f"  This checkpoint was created with a different network architecture.")
#             print(f"  Skipping and trying next checkpoint...")
#             continue

if not checkpoint_loaded:
    print("\n" + "="*60)
    print("STARTING FRESH TRAINING")
    print("="*60)
    print("No compatible checkpoint found. Starting from scratch.")
    print("Old checkpoints are incompatible due to architecture changes.")
    print("="*60 + "\n")

gamma = 0.99
epsilon_min = 0.1
epsilon_decay = 0.9995
batch_size = 32
target_update_freq = 5000
steps = 0

print(f"Training configuration:")
print(f"  Action size: {action_size}")
print(f"  Starting episode: {start_episode}")
print(f"  Gamma: {gamma}")
print(f"  Initial epsilon: {epsilon:.3f}")
print(f"  Epsilon decay: {epsilon_decay}")
print(f"  Epsilon min: {epsilon_min}")
print(f"  Batch size: {batch_size}")
print(f"  Target update frequency: {target_update_freq} steps")
print(f"  Replay buffer capacity: {replay_buffer.buffer.maxlen}")
print(f"  Rendering: {'ENABLED' if ENABLE_RENDERING else 'DISABLED'}")
if ENABLE_RENDERING:
    print(f"  Render frequency: Every {RENDER_EVERY_N_EPISODES} episode(s)")
print()

for episode in range(start_episode, 10000):
    # Determine if we should render this episode
    should_render = ENABLE_RENDERING and (episode % RENDER_EVERY_N_EPISODES == 0)
    
    # Reset environment - OLD GYM API (returns just state, not tuple)
    raw_state = env.reset()
    
    preprocessed_frame = preprocessor.preprocess(raw_state)
    state = frame_stack.reset(preprocessed_frame)
    state = torch.FloatTensor(state).to(device)
    
    total_reward = 0
    episode_steps = 0
    
    while True:
        # Render if enabled for this episode
        if should_render:
            env.render()
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state.unsqueeze(0))
                action = q_values.argmax().item()
        
        # Take action - OLD GYM API (returns 4 values, not 5)
        next_raw_state, reward, done, info = env.step(action)
        
        # Preprocess next state
        next_preprocessed_frame = preprocessor.preprocess(next_raw_state)
        next_state = frame_stack.step(next_preprocessed_frame)
        next_state = torch.FloatTensor(next_state).to(device)
        
        # Store in replay buffer
        replay_buffer.push(
            state.cpu().numpy(),
            action,
            reward,
            next_state.cpu().numpy(),
            done
        )
        
        # Train if enough samples
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states = torch.FloatTensor(np.array([s for s, _, _, _, _ in batch])).to(device)
            actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(device)
            rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(device)
            next_states = torch.FloatTensor(np.array([s for _, _, _, s, _ in batch])).to(device)
            dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(device)
            
            # Current Q values
            current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q values
            with torch.no_grad():
                max_next_q = target_network(next_states).max(1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)
            
            # Compute loss and update
            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
            optimizer.step()
        
        total_reward += reward
        state = next_state
        steps += 1
        episode_steps += 1
        
        # Update target network based on steps
        if steps % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
            print(f"  → Target network updated at step {steps}")
        
        # Decay epsilon after each step
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if done:
            break
    
    # Logging
    x_pos = info.get('x_pos', 0)
    render_indicator = "🎮" if should_render else "  "
    print(f"{render_indicator} Ep {episode:4d} | Steps: {episode_steps:4d} | Reward: {total_reward:6.0f} | X-pos: {x_pos:4d} | ε: {epsilon:.4f} | Buffer: {len(replay_buffer):6d}")
    
    # Save checkpoint periodically (always save latest)
    if episode > 0 and episode % 200 == 0:
        checkpoint_path = f'mario_dqn_episode_{episode}.pth'
        torch.save({
            'episode': episode,
            'q_network_state_dict': q_network.state_dict(),
            'target_network_state_dict': target_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
            'steps': steps,
        }, checkpoint_path)
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # Always save latest checkpoint
    if episode % 10 == 0:
        torch.save({
            'episode': episode,
            'q_network_state_dict': q_network.state_dict(),
            'target_network_state_dict': target_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
            'steps': steps,
        }, 'mario_dqn_latest.pth')

env.close()
print("Training complete!")