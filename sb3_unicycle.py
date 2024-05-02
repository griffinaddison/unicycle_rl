import gym

env = gym.make("LunarLander-v2")

env.reset()

for step in range(200):
    env.render()
    obs, reward, done, info = env.step(end.action_space.sample())
    print(reward)


env.close()
