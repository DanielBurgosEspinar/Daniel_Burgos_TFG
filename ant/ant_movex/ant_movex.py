import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# Crear directorios para modelos y logs
model_dir = "models_ant_v1"
log_dir = "logs_ant_v1"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

N_ENVS = 4
N_STEPS =4000000

def train(env):
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)

    
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
    

def test(env, path_to_model):
    model = PPO.load(path_to_model, env=env)
    
    num_episodes = 10
    total_reward = 0
    total_length = 0
    total_falls=0
    x_pos_total=0
    y_pos_total=0
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        env.render()

        ep_len = 0
        ep_reward = 0
        caido=False
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            if env.unwrapped.data.qpos[2]> 1 or env.unwrapped.data.qpos[2]<0.2:
                caido=True
            
            if terminated or truncated:
                # Obtener posición x y z del torso del ant
                x_pos = env.unwrapped.data.qpos[0]
                y_pos = abs(env.unwrapped.data.qpos[1])

                print(f"x: {x_pos:.3f}, y: {y_pos:.3f}")
                print(f"{ep_len=}  {ep_reward=}")
                if caido==True:
                    print("El modelo se ha caido")
                    total_falls+=1
                break

        total_length += ep_len
        total_reward += ep_reward
        x_pos_total+= x_pos
        y_pos_total+= y_pos


    print(f"Recompensa media: {total_reward / num_episodes}, Duracion de ep media: {total_length / num_episodes}")
    print(f"Distancia media en x: {x_pos_total / num_episodes}, Desviacion en y media: {y_pos_total / num_episodes}")
    print("Caidas totales:",total_falls)

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
        train(env)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        model_path = Path(args.model_path)
        env = gym.make("Ant-v5", render_mode='human')
        test(env,model_path)
