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
    def __init__(self, touch_rewards, shaping_rewards, max_steps=1000, n=7, use_potential_shaping=True):
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
        # shape = (317,)
        # but if we include the distance vector, it's 338 dimensional
        shape = (317+len(touch_rewards)+len(shaping_rewards),)
        # hand distances are only n, item distances and contacts are n*(n-1)/2, hence:
        if shaping_rewards is None:
            use_item_dist_rewards = False
            use_hand_dist_rewards = False
        elif shaping_rewards.shape[0] == n:
            use_item_dist_rewards = False
            use_hand_dist_rewards = True
        elif shaping_rewards.shape[0] == n*(n-1)//2:
            use_item_dist_rewards = True
            use_hand_dist_rewards = False
        elif shaping_rewards.shape[0] == n*(n+1)//2:
            use_item_dist_rewards = True
            use_hand_dist_rewards = True
        else:
            raise ValueError("shaping_rewards must have length n*(n-1)/2, n*(n+1)/2, n*(n-1), or n**2")
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self.touch_rewards = touch_rewards
        self.shaping_rewards = shaping_rewards
        self.max_steps = max_steps
        self.current_step = 0
        self.use_item_dist_rewards = use_item_dist_rewards
        self.use_hand_dist_rewards = use_hand_dist_rewards
        if use_potential_shaping:
            self.last_shaped_reward = 0
        self.use_potential_shaping = use_potential_shaping
        self.n = n

    def step(self, action):
        self.current_step += 1
        next_state = self.panda_instance.step(np.array(action), 
                                              inc_contact_array=True,
                                              inc_item_dists=self.use_item_dist_rewards, 
                                              inc_hand_dists=self.use_hand_dist_rewards)
        reward = self.compute_reward()
        # we return done = False because we don't have a termination condition besides max_steps
        done = self.is_done()
        truncated = self.is_done()
        if done and self.use_potential_shaping:
            # need to subtract last shaped reward or else it's actually rewarded for shaping at the end
            reward -= self.last_shaped_reward
        # we return 
        return next_state, reward, done, truncated, {}

    # apparently need to add **kwargs because stable_baselines3 passes it a seed
    # I don't think we need to do anything with the seed because we always want the same initial state
    def reset(self, **kwargs):
        self.current_step = 0
        self.panda_instance.reset()
        initial_state = np.concatenate([self.panda_instance.data.qpos, self.panda_instance.data.qvel])
        initial_state = np.concatenate([initial_state, self.panda_instance.make_contact_array()])
        if self.use_item_dist_rewards:
            # initial_state = np.concatenate([initial_state, self.panda_instance.make_distance_matrix(return_vec=True)])
            initial_state = np.concatenate([initial_state, self.panda_instance.make_item_distance_vector()])
        if self.use_hand_dist_rewards:
            # initial_state = np.concatenate([initial_state, self.panda_instance.make_hand_distance_matrix(return_vec=True)])
            initial_state = np.concatenate([initial_state, self.panda_instance.make_hand_distance_vector()])
        # also apparently need to return some "reset_infos" dict
        # maybe it can be None?
        if self.use_potential_shaping:
            self.last_shaped_reward = 0
        return initial_state, None

    def compute_reward(self):
        reward_features = self.panda_instance.make_contact_array()
        reward = np.dot(reward_features, self.touch_rewards)
        assert reward >= 0
        # compute shaped reward separately so that we can subtract it at the next step
        shaped_reward = 0
        if self.use_item_dist_rewards:
            item_distance_vec = self.panda_instance.make_item_distance_vector()
            clipped_distance_vec = np.clip(item_distance_vec, 0.01, 1000)
            inverse_distance_vec = 1 / clipped_distance_vec
            shaped_reward += np.dot(inverse_distance_vec, self.shaping_rewards[:(self.n*(self.n-1)//2)])
            assert shaped_reward >= 0
        if self.use_hand_dist_rewards:
            hand_distance_vec = self.panda_instance.make_hand_distance_vector()
            clipped_hand_distance_vec = np.clip(hand_distance_vec, 0.01, 1000)
            inverse_hand_distance_vec = 1 / clipped_hand_distance_vec
            hand_rewards = self.shaping_rewards if not self.use_item_dist_rewards else self.shaping_rewards[(self.n*(self.n-1)//2):]
            shaped_reward += np.dot(inverse_hand_distance_vec, hand_rewards)
            assert shaped_reward >= 0
        if self.use_potential_shaping:
            reward += shaped_reward - self.last_shaped_reward
            self.last_shaped_reward = shaped_reward
        else:
            reward += shaped_reward
        return reward

    def is_done(self):
        return self.current_step >= self.max_steps


# Example: EVERYTHING MUST TOUCH EVERYTHING
touch_rewards = np.ones(21)
shaping_rewards = np.ones(28)/100
shaping_rewards[21:] / 100

env = PandaEnv(touch_rewards, shaping_rewards, max_steps=3000, use_potential_shaping=False)

# first things first, wandb
wandb.init(project="panda-mujoco", sync_tensorboard=True)

# as mentioned below, use a random number just to keep things distinct
# actually, we'll go ahead and use directories for this:
# runs/{random_number}/*
# random_number = np.random.randint(100000)
# actually that's dumb, we should use the date/time instead
time = datetime.datetime.now()
# and actually let's add a custom message as well
# get a message from the user:
message = input("Enter a name for this run: ")
name = f"{message} {time}"
# first make runs/{name}, also making runs if it doesn't exist
os.makedirs(f"runs/{name}", exist_ok=True)
# then make runs/{name}/models and runs/{name}/images
os.makedirs(f"runs/{name}/models", exist_ok=True)
os.makedirs(f"runs/{name}/images", exist_ok=True)

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=f"runs/{name}/tensorboard/", 
    gamma=1,
)
# model = PPO.load("runs/2023-10-08 02:49:53.012073/models/model.zip", env=env, verbose=1, tensorboard_log=f"runs/{name}/tensorboard/", gamma=1)
# model.learn(total_timesteps=10000)
# actually we also want the learning to checkpoint every 100000 steps
# checkpoint_callback = CheckpointCallback(
#   save_freq=100000,
#   save_path=f"runs/{name}/models/",
#   name_prefix="panda_model",
#   save_replay_buffer=False,
#   save_vecnormalize=True,
# )

# we also want to integrate wandb
# here's the callback:
wandb_callback = WandbCallback(
    gradient_save_freq=100000,
    model_save_freq=100000,
    model_save_path=f"runs/{name}/models/",
    # model_save_replay_buffer=False,
    # model_save_vecnormalize=True,
    verbose=1
)

# model.learn(total_timesteps=5000000, callback=wandb_callback)
model.learn(total_timesteps=1000000, callback=wandb_callback)

obs, _ = env.reset()
for i in range(30000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if i % 100 == 0:
        print(action)
        print(reward)
        img_as_array = env.panda_instance.render()
        # Image.fromarray(img_as_array).save(f"test_{i}.png")
        # save to the appropriate directory
        Image.fromarray(img_as_array).save(f"runs/{name}/images/test_{i}.png")
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
model.save(f"runs/{name}/models/panda_model")
del model
model = PPO.load(f"runs/{name}/models/panda_model")