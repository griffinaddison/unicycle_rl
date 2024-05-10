# from stable_baselines3 import SAC
# from doq_quadruped_env import DoqQuadrupedEnv
# import cv2
# import imageio
#
# env = DoqQuadrupedEnv(render_mode="rgb_array")
# model = SAC.load("doq_quadruped_env.zip")
#
# obs, info = env.reset()
# frames = []
# for _ in range(500):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(action)
#     image = env.render()
#     if _ % 5 == 0:
#         frames.append(image)
#     cv2.imshow("image", image)
#     cv2.waitKey(100)
#     if done or truncated:
#         obs, info = env.reset()

# uncomment to save result as gif
# with imageio.get_writer("media/test.gif", mode="I") as writer:
#     for idx, frame in enumerate(frames):
#         writer.append_data(frame)
import gym
from stable_baselines3 import A2C, PPO, SAC
from euc_env import EucEnv
from stable_baselines3.common.env_checker import check_env
import os
import time
import argparse


def train():
    
    model_dir = f"models/SAC-{int(time.time())}"
    log_dir = f"logs/SAC-{int(time.time())}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # if not os.path.exists(models_dir):
    #     os.makedirs(models_dir)
    #
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)

    env = EucEnv(render_mode="rgb_array")
    check_env(env)
    env.reset()


    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    TIMESTEPS = 10000

    # Every 10k timesteps, save the model
    for i in range(1,100):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="SAC")
        model.save(f"{model_dir}/{TIMESTEPS*i}")

    #
    # vec_env = model.get_env()
    # obs = vec_env.reset()




    env.close()


# env = gym.make("LunarLander-v2", render_mode="rgb_array")

def test(model_path):
    env = EucEnv(render_mode="human")
    env.reset()

    # models_dir = "models/SAC-1714768282"
    # model_path = f"{models_dir}/980000.zip"

    model = SAC.load(model_path, env=env)

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


    env.close()


if __name__ == '__main__':

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    # parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()


    if args.train:
        # gymenv = gym.make(args.gymenv, render_mode=None)
        train()

    if(args.test):
        if os.path.isfile(args.test):
            test(model_path=args.test)
        else:
            print(f'{args.test} not found.')
