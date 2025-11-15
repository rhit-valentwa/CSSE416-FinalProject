# CSSE416-FinalProject
This repository serves as the primary storage location for all code relating to the CSSE416: Deep Learning final project "Super Learning Bros".

## Important Notes
- We included the source code for <https://github.com/justinmeister/Mario-Level-1>. We did *not* comment this code as this is not code we created or modified (with the exception of fixing a bug with how mario jumped).
- We are assuming you are testing the code on a Rose-Hulman Laptop and not a Linux server (like Gebru) or a Mac device. As such our code and instructions have been designed with Windows 11 in mind (and treat Linux as more of a second-class citizen, we never tested any code on MacOS).
- Generative AI is utilized in the creation of this code.
	- Claude Sonnet 4.5 provided advice and some implementation details surrounding DDQN and PPO, and assisted with debugging some annoying API details with OpenAI's Gymnasium
	- All developers had GitHub CoPilot enabled, while most the of the changes it recommended were ignored, a few details such as formatting, and variable naming were often accepted. 
- There are two main files which run basically all of the logic for this project. 
	- CSSE416-FINALPROJECT/main.py
		- This file contains the actual RL algorithms (in the main repo just DDQN)
	- CSSE416-FINALPROJECT/gymnasium_env/envs/mario_world.py
		- This file provides the connections between OpenAI's Gymnasium and the Mario level 1-1 implementation

### Overview of Repo's Current Features
- PyTorch-based DQN agent with target network and experience replay
- Custom environment using Gymnasium API
- MultiBinary action mapping for Mario controls
- Configurable hyperparameters for training
- Replay buffer and epsilon-greedy exploration
- Model checkpointing and reward logging (commented for easy activation)

### Usage (Tested on Python 3.12, will likely work on different versions of Python)
1. Ensure dependencies are installed (see `requirements.txt` in `mario_game/`).
	a. We would highly recommend installing the CUDA verison of torch as we make extensive use of GPU optimizations
	b. You may install CUDA torch using this command `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
	c. You can do this by running `pip install -r requirements.txt` (if you chose to install torch with CUDA support, then that should override this default torch install)
2. Run the training and display script (run from the top-level repo folder CSSE416-FinalProject):
	`python main.py`
3. Training progress and rewards are printed to the console.
4. A window will appear showing Mario's actual progress in the level.

### Key Hyperparameters (in `main.py`)
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
- The script uses `MarioLevelEnv` from `gymnasium_env/envs/mario_world.py`.
- See environment code for details on observation and action spaces.

### Customization
- Adjust hyperparameters at the top of `main.py` to tune training.
- Modify the environment or agent for advanced RL experiments.