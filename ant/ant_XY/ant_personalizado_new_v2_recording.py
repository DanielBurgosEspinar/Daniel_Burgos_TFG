import gymnasium as gym
import numpy as np
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import argparse
from pathlib import Path
from tqdm import tqdm

# Wrapper para recompensar acercarse al objetivo (x, y)
class AntGoToXYWrapper(gym.Wrapper):
    def __init__(self, env, target_x, target_y):
        super().__init__(env)
        self.target_x = target_x
        self.target_y = target_y
        self.prev_distance = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        x = info.get("x_position", 0.0)
        y = info.get("y_position", 0.0)
        self.prev_distance = np.sqrt((x - self.target_x) ** 2 + (y - self.target_y) ** 2)
        return obs, info

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)

        x = info.get("x_position", 0.0)
        y = info.get("y_position", 0.0)
        distance = np.sqrt((x - self.target_x) ** 2 + (y - self.target_y) ** 2)

        # Recompensa por progreso hacia el objetivo
        progress_reward = max(self.prev_distance - distance, -1.0)
        self.prev_distance = distance

        # Recompensa adicional por estar muy cerca
        proximity_reward = progress_reward - 0.2 * distance # se duplica la penalizacion por estar lejos

        if distance < 1:
            proximity_reward += 10.0
        #custom_reward *= 5.0

        if distance < 0.1: #mas resstrictivo el umbral
            proximity_reward += 500
            print("ha llegado")
            done=True

        # Sumar a la recompensa base (sin forward reward)
        total_reward = base_reward + proximity_reward

        return obs, total_reward, done, truncated, info

# Configuración de directorios
model_dir = "models_ant_XY"
log_dir = "logs_ant_XY"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

N_ENVS=4
N_STEPS=4000000


def make_custom_ant_env():
    def _init():
        env = gym.make(
            "Ant-v5",
            forward_reward_weight=0.0,
            ctrl_cost_weight=0.5,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            exclude_current_positions_from_observation=False
        )
        env = AntGoToXYWrapper(env, target_x=-5, target_y=-5)
        return env
    return _init


def train():
    
    env = make_vec_env(make_custom_ant_env(), n_envs=4, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=log_dir, learning_rate=2.5e-4)

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{train_time}"
    model_path = f"{model_dir}/{run_name}"

    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_path,
        log_path=log_dir,
        eval_freq=25000,  # Evaluar cada N timesteps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )


    timesteps = N_STEPS

    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=eval_callback,
    )

    # Save final model
    model.save(f"{model_path}/final_model")

def test_and_record(path_to_model, video_folder="recordings", num_episodes=1):
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(
        "Ant-v5",
        forward_reward_weight=0.0,
        exclude_current_positions_from_observation=False,
        render_mode='rgb_array'
    )
    target_x = 5
    target_y = 5
    env = AntGoToXYWrapper(env, target_x, target_y)

    # Añadir wrapper para grabación
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="ant_test"
    )

    model = PPO.load(path_to_model, env=env)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    print(f"Grabación de {num_episodes} episodios guardada en '{video_folder}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )

    args = parser.parse_args()
    if args.run == "train":
        env = make_vec_env("Ant-v5", n_envs=N_ENVS,vec_env_cls=SubprocVecEnv)
        train()
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        model_path = Path(args.model_path)
        #env = gym.make("Ant-v5", render_mode='human')
        test_and_record(model_path)

