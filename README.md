# CSSE416-FinalProject
This repository serves as the primary storage location for all code relating to the CSSE416: Deep Learning final project "Super Learning Bros".
  
## Main DQN Training Script (`main.py`)

This file implements a Deep Q-Network (DQN) agent to train on the custom Mario environment (`MarioLevelEnv`).

### Features
- PyTorch-based DQN agent with target network and experience replay
- Custom environment using Gymnasium API
- MultiBinary action mapping for Mario controls
- Configurable hyperparameters for training
- Replay buffer and epsilon-greedy exploration
- Model checkpointing and reward logging (commented for easy activation)

### Usage
1. Ensure dependencies are installed (see `requirements.txt` in `mario_game/`).
2. Run the training script:
	```powershell
	python main.py
	```
3. Training progress and rewards are printed to the console.

### Key Hyperparameters
- `NUMBER_OF_SEQUENTIAL_FRAMES`: Number of stacked frames for state input
- `REPLAY_BUFFER_SIZE`: Size of experience replay buffer
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Optimizer learning rate
- `GAMMA`: Discount factor for future rewards
- `EPSILON_START`, `EPSILON_DECAY`, `EPSILON_MIN`: Exploration parameters

### Checkpoints & Logging
- Model checkpoints and reward logs can be enabled by uncommenting relevant lines in the script.
- Checkpoints are saved in the `checkpoints/` directory.

### Environment
- The script uses `MarioLevelEnv` from `gymnasium_env.envs.mario_world`.
- See environment code for details on observation and action spaces.

### Customization
- Adjust hyperparameters at the top of `main.py` to tune training.
- Modify the environment or agent for advanced RL experiments.

---
For more details, see comments in `main.py` and the environment code in `gymnasium_env/envs/mario_world.py`.
