import gym
from stable_baselines3 import A2C, PPO
import os


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2", render_mode="rgb_array")
env.reset()


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")



env.close()
