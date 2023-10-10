from panda import Panda
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image
import os
import datetime
import wandb
from wandb.integration.sb3 import WandbCallback

class PandaEnv(gym.Env):
    def __init__(self, touch_rewards, max_steps=1000):
        """
        :param touch_rewards: A NumPy array representing rewards and penalties for touching each object pair.
        :param max_steps: The maximum number of steps before the environment terminates.
        """
        self.panda_instance = Panda()
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(8,), dtype=np.float32)
        # using these lines:
        # initial_state = np.concatenate([self.panda_instance.data.qpos, self.panda_instance.data.qvel])
        # breakpoint()
        # we figure out that the state shape is 317 dimensional
        shape = (317,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self.touch_rewards = np.array(touch_rewards)
        self.max_steps = max_steps
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        next_state = self.panda_instance.step(np.array(action))
        reward = self.compute_reward()
        # we return done = False because we don't have a termination condition besides max_steps
        done = self.is_done()
        truncated = self.is_done()
        # we return 
        return next_state, reward, done, truncated, {}

    # apparently need to add **kwargs because stable_baselines3 passes it a seed
    # I don't think we need to do anything with the seed because we always want the same initial state
    def reset(self, **kwargs):
        self.current_step = 0
        self.panda_instance.reset()
        initial_state = np.concatenate([self.panda_instance.data.qpos, self.panda_instance.data.qvel])
        # also apparently need to return some "reset_infos" dict
        # maybe it can be None?
        return initial_state, None

    def compute_reward(self):
        contact_array = self.panda_instance.make_contact_array()
        reward = np.dot(contact_array, self.touch_rewards)
        return reward

    def is_done(self):
        return self.current_step >= self.max_steps

# first things first, wandb
wandb.init(project="panda-mujoco", sync_tensorboard=True)

# Example: EVERYTHING MUST TOUCH EVERYTHING
touch_rewards = np.ones(21)

env = PandaEnv(touch_rewards, max_steps=10000)

# as mentioned below, use a random number just to keep things distinct
# actually, we'll go ahead and use directories for this:
# runs/{random_number}/*
# random_number = np.random.randint(100000)
# actually that's dumb, we should use the date/time instead
time = datetime.datetime.now()
# first make runs/{time}, also making runs if it doesn't exist
os.makedirs(f"runs/{time}", exist_ok=True)
# then make runs/{time}/models and runs/{time}/images
os.makedirs(f"runs/{time}/models", exist_ok=True)
os.makedirs(f"runs/{time}/images", exist_ok=True)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{time}/tensorboard/")
# model.learn(total_timesteps=10000)
# actually we also want the learning to checkpoint every 100000 steps
# checkpoint_callback = CheckpointCallback(
#   save_freq=100000,
#   save_path=f"runs/{time}/models/",
#   name_prefix="panda_model",
#   save_replay_buffer=False,
#   save_vecnormalize=True,
# )

# we also want to integrate wandb
# here's the callback:
wandb_callback = WandbCallback(
    gradient_save_freq=100000,
    model_save_freq=100000,
    model_save_path=f"runs/{time}/models/",
    # model_save_replay_buffer=False,
    # model_save_vecnormalize=True,
    verbose=1
)

model.learn(total_timesteps=10000000, callback=wandb_callback)

obs, _ = env.reset()
for i in range(100000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if i % 1000 == 0:
        print(action)
        print(reward)
        img_as_array = env.panda_instance.render()
        # Image.fromarray(img_as_array).save(f"test_{i}.png")
        # save to the appropriate directory
        Image.fromarray(img_as_array).save(f"runs/{time}/images/test_{i}.png")
    if done or truncated:
        obs, _ = env.reset()

# save and reload just to make sure it works
# also generate a random number just to make sure we don't overwrite the model
# model.save("panda_model")
# del model
# model = PPO.load("panda_model")
# model.save(f"panda_model_{time}")
# del model
# model = PPO.load(f"panda_model_{time}")
# save to the appropriate directory
model.save(f"runs/{time}/models/panda_model")
del model
model = PPO.load(f"runs/{time}/models/panda_model")