import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.env_checker import check_env
import os
import time
import argparse

from euc_env import EucEnv
from wbc_wrapper import WbcWrapper

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

    ## Wrap base environment with wbc wrapper
    env = WbcWrapper(env)


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
    env = WbcWrapper(env)

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

        count = 0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render("human")
            # if count % 10 == 0:
                # print(f"\n reward: {reward}")
                # print(f"\n vel_rew: {info['velocity_reward']}")
            #     print(f"\n reward: {reward}, vel_rew: {info['velocity_reward']}, orient_rew: {info['orientation_reward']}, control_cost: {info['control_cost']}")
            count+=1


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
