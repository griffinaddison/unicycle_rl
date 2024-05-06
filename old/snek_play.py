import gym
from stable_baselines3 import A2C, PPO
import os
from snakeenv import SnekEnv
import time as time

# env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = SnekEnv()
env.reset()

models_dir = "models/1714680742"
model_path = f"{models_dir}/70000.zip"

model = PPO.load(model_path, env=env)

vec_env = model.get_env()
obs = vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")
#
#

episodes = 10

for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        time.sleep(0.1)


env.close()
