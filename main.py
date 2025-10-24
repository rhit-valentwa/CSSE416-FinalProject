from gymnasium_env.envs.mario_world import MarioLevelEnv

env = MarioLevelEnv(render_mode="human")
num_episodes = 25

for ep in range(1, num_episodes + 1):
    obs, info = env.reset()
    done = False
    trunc = False
    ep_return = 0.0

    while not (done or trunc):
        action = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(action)
        ep_return += rew

    print(f"Episode {ep}: total reward = {ep_return:.2f}")
env.close()