
import gym
from stable_baselines3 import  PPO
import os
import time
from snakeenv import SnekEnv


models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = SnekEnv()
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
# Every 10k timesteps, save the model
for i in range(1,1000000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


vec_env = model.get_env()
obs = vec_env.reset()




env.close()
