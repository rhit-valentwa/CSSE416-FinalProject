import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from gymnasium_env.envs.mario_world import MarioLevelEnv
import os
import shutil
from PIL import Image

# =============================
# CONSTANTS & HYPERPARAMETERS
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_DEVICE = True
NUMBER_OF_SEQUENTIAL_FRAMES = 6
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0  # Start with full exploration
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05  # Allow exploration even late in training
TAU = 0.001  # Slightly slower soft updates
N_EPISODES = 10000
REWARD_HISTORY_SIZE = 100
CHECKPOINT_FREQ = 500
LOG_REWARD_DIR = "logs/rew"
CHECKPOINT_DIR = "checkpoints"
BUFFER_SAVE_DIR = "cnn_analyze/game_frames"
CNN_DATA_DIR = "cnn_analyze/cnn_frames"
ACTION_SIZE = 8
MIN_REPLAY_SIZE = 1000  # Don't train until we have enough diverse experiences
MAX_BUFFER_BATCHES = 5  # Keep only the 5 most recent buffer batches

if PRINT_DEVICE:
    print(f"Using device: {DEVICE}")

# Deep Q-Network with hooks for data capture
class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUMBER_OF_SEQUENTIAL_FRAMES, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Automatically calculate flattened size
        conv_out_size = self._get_conv_out((NUMBER_OF_SEQUENTIAL_FRAMES, 60, 80))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        
        # Storage for intermediate activations
        self.activations = {}
        self.register_hooks()
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def register_hooks(self):
        """Register forward hooks to capture intermediate layer outputs"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for each conv layer
        self.conv[0].register_forward_hook(get_activation('conv1'))
        self.conv[2].register_forward_hook(get_activation('conv2'))
        self.conv[4].register_forward_hook(get_activation('conv3'))
    
    def forward(self, x):
        # Store input
        self.activations['input'] = x.detach()
        
        conv_out = self.conv(x)
        
        # Store pre-flatten output
        self.activations['conv_out'] = conv_out.detach()
        
        conv_out = conv_out.view(x.size(0), -1)  # Flatten
        fc_out = self.fc(conv_out)
        
        # Store final Q-values
        self.activations['q_values'] = fc_out.detach()
        
        return fc_out
    
    def get_activations(self):
        """Return a copy of all stored activations"""
        return {k: v.clone() for k, v in self.activations.items()}

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
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


# =============================
# CNN Data Saver
# =============================
class CNNDataSaver:
    def __init__(self, base_dir=CNN_DATA_DIR, max_batches=MAX_BUFFER_BATCHES):
        self.base_dir = base_dir
        self.max_batches = max_batches
        self.batch_counter = 0
        os.makedirs(base_dir, exist_ok=True)
    
    def save_cnn_data(self, activations, episode, step, action_taken, q_values):
        """Save CNN layer activations as numpy arrays"""
        batch_name = f"batch_{self.batch_counter:04d}_ep{episode}_step{step}"
        batch_dir = os.path.join(self.base_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)
        
        # Save all activations as numpy arrays
        for layer_name, activation in activations.items():
            activation_np = activation.cpu().numpy()
            save_path = os.path.join(batch_dir, f'{layer_name}.npy')
            np.save(save_path, activation_np)
        
        # Save action and Q-value info
        info_data = {
            'action_taken': action_taken,
            'q_values': q_values.cpu().numpy(),
            'episode': episode,
            'step': step
        }
        np.save(os.path.join(batch_dir, 'info.npy'), info_data)
        
        self.batch_counter += 1
        self._cleanup_old_batches()
    
    def _cleanup_old_batches(self):
        """Remove old batches, keeping only the most recent max_batches"""
        batch_dirs = [d for d in os.listdir(self.base_dir) 
                     if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith('batch_')]
        
        batch_dirs.sort(key=lambda x: int(x.split('_')[1]))
        
        while len(batch_dirs) > self.max_batches:
            old_batch = batch_dirs.pop(0)
            old_batch_path = os.path.join(self.base_dir, old_batch)
            shutil.rmtree(old_batch_path)
            print(f"Removed old CNN data batch: {old_batch}")


# =============================
# Buffer Storage Functions
# =============================
class BufferManager:
    def __init__(self, base_dir=BUFFER_SAVE_DIR, max_batches=MAX_BUFFER_BATCHES):
        self.base_dir = base_dir
        self.max_batches = max_batches
        self.batch_counter = 0
        os.makedirs(base_dir, exist_ok=True)
    
    def save_buffers(self, env, episode, step):
        """Save the current state of all buffers from the environment"""
        batch_name = f"batch_{self.batch_counter:04d}_ep{episode}_step{step}"
        batch_dir = os.path.join(self.base_dir, batch_name)
        os.makedirs(batch_dir, exist_ok=True)
        
        buffer_types = [
            ('original', env.buffer_original),
            ('normalized', env.buffer_normalized),
            ('downscaled', env.buffer_downscaled),
            ('grayscale', env.buffer_grayscale)
        ]
        
        for buffer_name, buffer in buffer_types:
            buffer_subdir = os.path.join(batch_dir, buffer_name)
            os.makedirs(buffer_subdir, exist_ok=True)
            
            for frame_idx, frame in enumerate(buffer):
                self._save_frame(frame, buffer_subdir, frame_idx, buffer_name)
        
        self.batch_counter += 1
        self._cleanup_old_batches()
    
    def _save_frame(self, frame, save_dir, frame_idx, buffer_type):
        """Save a single frame as an image"""
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        if len(frame.shape) == 3:
            if frame.shape[0] == 1:
                frame = frame[0]
            elif frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
        
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        img = Image.fromarray(frame)
        filepath = os.path.join(save_dir, f"frame_{frame_idx:02d}.png")
        img.save(filepath)
    
    def _cleanup_old_batches(self):
        """Remove old batches, keeping only the most recent max_batches"""
        batch_dirs = [d for d in os.listdir(self.base_dir) 
                     if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith('batch_')]
        
        batch_dirs.sort(key=lambda x: int(x.split('_')[1]))
        
        while len(batch_dirs) > self.max_batches:
            old_batch = batch_dirs.pop(0)
            old_batch_path = os.path.join(self.base_dir, old_batch)
            shutil.rmtree(old_batch_path)
            print(f"Removed old buffer batch: {old_batch}")


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

    def select_action(self, state, env, capture_data=False):
        if random.random() < self.epsilon:
            action_idx = env.action_space.sample()
            action_idx_int = multibinary_to_index(action_idx)
            activations = None
            q_values = None
        else:
            with torch.no_grad():
                q_values = self.q_network(state.unsqueeze(0))
                action_idx_int = q_values.argmax().item()
                action_idx = index_to_multibinary(action_idx_int)
                
                # Get activations if data capture is requested
                if capture_data:
                    activations = self.q_network.get_activations()
                else:
                    activations = None
        
        return action_idx, action_idx_int, activations, q_values

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
        """Load a saved checkpoint into the agent's networks and optimizer."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)

# =============================
# Training Loop
# =============================
env = MarioLevelEnv(render_mode="human", number_of_sequential_frames=NUMBER_OF_SEQUENTIAL_FRAMES)
agent = DQNAgent(ACTION_SIZE, DEVICE)
agent.load('checkpoints/solid_saves/better_4000.pth')

# Initialize managers
buffer_manager = BufferManager()
cnn_data_saver = CNNDataSaver()

reward_history = deque(maxlen=REWARD_HISTORY_SIZE)

for episode in range(N_EPISODES):
    state, _ = env.reset()
    state = torch.FloatTensor(state).to(DEVICE)
    total_reward = 0
    steps_in_episode = 0
    
    while True:
        steps_in_episode += 1
        
        # Capture CNN data every 100 steps
        capture_cnn = (steps_in_episode % 100 == 0)
        
        action, action_idx, activations, q_values = agent.select_action(state, env, capture_data=capture_cnn)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        
        agent.store_transition(state, action, reward, next_state, done)
        
        # Save buffers and CNN data every 100 steps
        if steps_in_episode % 100 == 0:
            buffer_manager.save_buffers(env, episode, steps_in_episode)
            
            if activations is not None and q_values is not None:
                cnn_data_saver.save_cnn_data(activations, episode, steps_in_episode, 
                                            action_idx, q_values)
        
        # Only train if we have enough experiences
        # if len(agent.replay_buffer) >= MIN_REPLAY_SIZE:
        #     agent.train_step()
        #     agent.update_target_network()
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    agent.decay_epsilon()
    reward_history.append(total_reward)
    
    print(f"Episode {episode}: Reward={total_reward:.2f}, Steps={steps_in_episode}, Epsilon={agent.epsilon:.4f}")
    
    if len(reward_history) == REWARD_HISTORY_SIZE:
        avg_reward = sum(reward_history)/REWARD_HISTORY_SIZE
        print(f"Episode {episode}; Average Reward: {avg_reward:.2f}")
        
        os.makedirs(LOG_REWARD_DIR, exist_ok=True)
        log_path = os.path.join(LOG_REWARD_DIR, "avg_reward.txt")
        with open(log_path, "a") as f:
            f.write(f"Episode {episode}; Average Reward: {avg_reward:.2f}\n")
    
    if episode % 10 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    if episode % CHECKPOINT_FREQ == 0 and episode > 0:
        agent.save(episode)