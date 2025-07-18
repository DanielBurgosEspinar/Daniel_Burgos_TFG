import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env


from go1_mujoco_env_goXY_v4 import Go1MujocoEnv


from tqdm import tqdm
import numpy as np

import torch

MODEL_DIR = "models_v4"
LOG_DIR = "logs_v4"

#se ejecuta con python train.py --run train
def train(args):
    vec_env = make_vec_env(
        Go1MujocoEnv,
        env_kwargs={"ctrl_type": args.ctrl_type},
        n_envs=args.num_parallel_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is None:
        run_name = f"{train_time}"
    else:
        run_name = f"{train_time}-{args.run_name}"

    model_path = f"{MODEL_DIR}/{run_name}"
    print(
        f"Training on {args.num_parallel_envs} parallel training environments and saving models to '{model_path}'"
    )

   # Evalua cada 5 ep y guarda el mejor modelo
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=model_path,
        log_path=LOG_DIR,
        eval_freq=args.eval_frequency,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, env=vec_env, verbose=1, tensorboard_log=LOG_DIR, device='cpu',
        )
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR ,device='cpu')

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=eval_callback,
    )
    # Save final model
    model.save(f"{model_path}/final_model")


def test(args):
    model_path = Path(args.model_path)

    
    # Render the episodes live
    env = Go1MujocoEnv(
        ctrl_type=args.ctrl_type,
        render_mode="human",
        
    )
    inter_frame_sleep = 0.016 #velocidad de simulacion incvial
    
    

    model = PPO.load(path=model_path, env=env, verbose=1)

    num_episodes = args.num_test_episodes
    total_reward = 0
    total_length = 0
    total_falls=0
    x_dist=0
    y_dist=0
    total_targets=0
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        env.render()

        ep_len = 0
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            
            time.sleep(inter_frame_sleep)

            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                total_falls += env._unhealthy_counter
                
                x_dist += env.data.qpos[0]
                y_dist += env.data.qpos[1]
                if np.linalg.norm(env.data.qpos[:2] - env._target_position) < env.distance_umbral:
                    print("llega")
                    total_targets+=1
                break

        total_length += ep_len
        total_reward += ep_reward

    print(f"Avg episode reward: {total_reward / num_episodes}, avg episode length: {total_length / num_episodes}")
    print(f"Total falls:",total_falls)
    print(f"Total targets:",total_targets)
    print(f"Average dist in x:", {x_dist/num_episodes})
    print(f"Average dist in y:", {y_dist/num_episodes})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=12,
        help="Number of parallel environments while training",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=10,
        help="Number of episodes to test the model",
    )
   
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=8_000_000,
        help="Number of timesteps to train the model for",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10_000,
        help="The frequency of evaluating the models while training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["torque", "position"],
        default="position",
        help="Whether the model should control the robot using torque or position control.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        test(args)