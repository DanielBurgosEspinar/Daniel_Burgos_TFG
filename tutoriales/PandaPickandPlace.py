import os
import mujoco

import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C,PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login
import numpy as np


env_id = "PandaPickAndPlace-v3"
env = make_vec_env(lambda: gym.make(env_id, reward_type="dense"), n_envs=4)
#env = make_vec_env(env_id, n_envs=4,reward_type="dense")
print(env.action_space)
obs = env.reset()
print("#################################################")
obs = env.reset()
print(obs["observation"])

print(env.envs[0].spec.max_episode_steps)

print("##########MUJOCO version")
print(mujoco.__version__)




# 3
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# 4
model = A2C(policy = "MultiInputPolicy",
            env = env,
            verbose=1)
# 5
import time

start_time = time.time()
model.learn(2_000_000)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Tiempo de entrenamiento: {elapsed_time:.2f} segundos")
print(f"Tiempo de entrenamiento: {elapsed_time / 60:.2f} minutos")

# 6
model_name = "a2c-PandaPickAndPlace-v3";
model.save(model_name)
env.save("a2c-vec_normalize.pkl")

# 7
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("PandaPickAndPlace-v3",reward_type="dense")])
eval_env = VecNormalize.load("a2c-vec_normalize.pkl", eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load(model_name)

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")


