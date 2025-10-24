from gymnasium_env.envs.mario_world import MarioLevelEnv

class Agent:
    def run(self):
        env = MarioLevelEnv(render_mode="human")
        num_episodes = 25
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []

        policy_dqn = DQN(state_size=num_states, action_size=num_actions)

        for ep in range(1, num_episodes + 1):
            obs, info = env.reset()
            done = False
            trunc = False
            ep_reward = 0.0

            while not (done or trunc):
                action = env.action_space.sample()
                obs, reward, done, trunc, info = env.step(action)
                ep_reward += reward

            print(f"Episode {ep}: total reward = {ep_reward:.2f}")
            rewards_per_episode.append(ep_reward)
        env.close()

if __name__ == "__main__":
    agent = Agent()
    agent.run()