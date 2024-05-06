from snakeenv import SnekEnv


env = SnekEnv()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while not done:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, done, trunctated, info = env.step(random_action)
		print('reward',reward)
