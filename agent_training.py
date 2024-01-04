from panda import Panda
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image
import os
import datetime
import wandb
from wandb.integration.sb3 import WandbCallback
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run Panda Environment with specified parameters.')
    parser.add_argument('--algorithm', choices=['PPO', 'SAC', 'TQC'], default='PPO',
                        help='SB3 algorithm to use: PPO, SAC, or TQC. Default is PPO.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for the SB3 algorithm. Default is 0.99.')
    parser.add_argument('--use-touch-rewards', action='store_true',
                        help='Include touch rewards in the environment.')
    parser.add_argument('--no-use-touch-rewards', action='store_false',
                        help='Do not include touch rewards in the environment.')
    parser.add_argument('--use-shaping-rewards', action='store_true',
                        help='Include shaping rewards in the environment.')
    parser.add_argument('--no-use-shaping-rewards', action='store_false',
                        help='Do not include shaping rewards in the environment.')
    parser.add_argument('--destination-pos', type=str, default='none',
                        help='Destination position as a comma-separated list of three values or "none". Default is "none".')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum number of steps in the environment. Default is 1000.')
    parser.add_argument('--learning-steps', type=int, default=1000000,
                        help='Number of learning steps. Default is 1000000.')
    parser.add_argument('--center-rewards', action='store_true', default=False,
                        help='Center rewards around 0 in the starting state. Default is False.')
    parser.add_argument('--test-steps', type=int, default=30000,
                        help='Number of steps to test the model. Default is 30000.')
    parser.add_argument('--test-render-interval', type=int, default=100,
                        help='Number of steps between renders during testing. Default is 100.')
    parser.add_argument('--use-potential-shaping', action='store_true', default=False,
                        help='Use potential function shaping rather than absolute shaping. Default is False.')
    parser.add_argument('--starting-position', type=str, default='default',
                        help='Starting position for the arm. Either "default", "ground" or "random_object".')

    # Parse the arguments
    args = parser.parse_args()
    return args

class PandaEnv(gym.Env):
    def __init__(self, touch_rewards, shaping_rewards, destination_pos=None, max_steps=1000, n=6, use_potential_shaping=True, center_rewards=False, start_pos="default"):
        """
        :param touch_rewards: A NumPy array representing rewards and penalties for touching each object pair.
        :param max_steps: The maximum number of steps before the environment terminates.
        """
        self.panda_instance = Panda()
        self.action_space = spaces.Box(low=-3, high=3, shape=(7,), dtype=np.float32)
        # using these lines:
        # initial_state = np.concatenate([self.panda_instance.data.qpos, self.panda_instance.data.qvel])
        # breakpoint()
        # we figure out that the state shape is 208 dimensional
        # shape = (208,)
        # but if we include the distance vector, it's ??? dimensional
        shape = [208]
        if touch_rewards is not None:
            shape[0] += len(touch_rewards)
        if shaping_rewards is not None:
            shape[0] += len(shaping_rewards)
        # shape = (208+len(touch_rewards)+len(shaping_rewards),)
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
        # ChatGPT says that using np.inf for the bounds can cause problems,
        # so we're just going to hope that -100 and 100 captures the actual range
        # fairly well. If not, there will be some clipping, but that's probably okay
        self.observation_space = spaces.Box(low=-100, high=100, shape=shape, dtype=np.float32)
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
        self.destination_pos = destination_pos
        self.center_rewards = center_rewards
        self.starting_reward = 0 # since compute_reward() uses self.starting_reward
        self.starting_reward = self.compute_reward()
        self.start_pos = start_pos
        self.move_to_start_pos()

    def move_to_start_pos(self):
        ground_z = 0.0  # Define the ground level Z-coordinate
        max_attempts = 100  # To prevent infinite loops
        attempts = 0

        if self.start_pos == "default":
            return
        elif self.start_pos == "ground":
            while attempts < max_attempts:
                # Fetch the current position of the end effector
                current_ee_pos = self.panda_instance.data.xpos[self.panda_instance.body_name_to_idx["ee_ref_container"]]
                # Check if the end effector is close enough to the ground
                if abs(current_ee_pos[2] - ground_z) < 0.3:
                    break
                else:
                    print(self.panda_instance.data.xpos[self.panda_instance.body_name_to_idx["ee_ref_container"]])
                # Apply a downward action
                action = np.zeros(7)
                action[5] -= 3  # Downward z movement
                self.step(action)
                attempts += 1
            return
        elif self.start_pos == "object":
            max_steps_per_stage = 200
            position_threshold = 0.1
            step_size = 0.01
        # 1. Choose a random object
            object_name = np.random.choice(Panda.item_names)
            object_idx = self.panda_instance.body_name_to_idx[object_name]
            print(f"Target Object: {object_name}")

            # 2. Move the end effector above the chosen object
            # print("Stage: Moving above the object")
            object_pos = self.panda_instance.data.xpos[object_idx]
            # above_object_pos = object_pos + np.array([0, 0, 0.4])  # Adjust Z-coordinate
            # step = 0
            hand_pos = self.panda_instance.data.xpos[self.panda_instance.body_name_to_idx["ee_ref_container"]]
            # # while np.linalg.norm(self.panda_instance.body_name_to_idx["ee_ref_container"] - above_object_pos) > position_threshold and step < max_steps_per_stage:
            # while np.linalg.norm(hand_pos - above_object_pos) > position_threshold and step < max_steps_per_stage:
            #     # Apply forces to move towards above_object_pos
            #     direction_to_target = object_pos - hand_pos
            #     # direction_to_target_normalized = direction_to_target / np.linalg.norm(direction_to_target)
            #     action = np.zeros(7)  # 7 control dimensions, adjust if different
            #     action[3:6] = direction_to_target * step_size # Apply movement in the 4th, 5th, 6th actuators
            #     action[6] = 255 # open the gripper
            #     self.panda_instance.step(action)
            #     if step % 10 == 0:
            #         # self.render()
            #         img = self.panda_instance.render()
            #         # save to "object_start/{object_name}/{stage}/{step}.png"
            #         # first make the directory
            #         os.makedirs(f"object_start/{object_name}/2", exist_ok=True)
            #         # then save the image
            #         Image.fromarray(img).save(f"object_start/{object_name}/2/{step}.png")
            #     step += 1

            # 3. Move the hand down to the object position
            print("Stage: Moving down to object")
            step = 0
            while np.linalg.norm(self.panda_instance.data.xpos - object_pos) > position_threshold and step < max_steps_per_stage:
                # Apply forces to move towards object_pos
                # self.step(move_down_action)
                direction_to_target = object_pos - hand_pos
                # direction_to_target_normalized = direction_to_target / np.linalg.norm(direction_to_target)
                action = np.zeros(7)  # 7 control dimensions, adjust if different
                action[3:6] = direction_to_target * step_size # Apply movement in the 4th, 5th, 6th actuators
                action[6] = 255 # keep the gripper open
                self.panda_instance.step(action)
                if step % 10 == 0:
                    # self.render()
                    img = self.panda_instance.render()
                    # save to "object_start/{object_name}/{stage}/{step}.png"
                    # first make the directory
                    os.makedirs(f"object_start/{object_name}/3", exist_ok=True)
                    # then save the image
                    Image.fromarray(img).save(f"object_start/{object_name}/3/{step}.png")
                step += 1

            # let's also save the final position data
            # first make the directory
            os.makedirs(f"object_start/{object_name}/qpos", exist_ok=True)
            # then save the data
            # np.save(f"object_start/{object_name}/3/final_pos.npy", self.panda_instance.data.xpos)
            # np.save(f"object_start/{object_name}/3/final_quat.npy", self.panda_instance.data.xquat)
            # chatgpt says I can just save qpos
            np.save(f"object_start/{object_name}/qpos/final.npy", self.panda_instance.data.qpos)


            # 4. Close the gripper
            # print("Stage: Closing gripper")
            # step = 0
            # while not self.check_contact_with_object() and step < max_steps_per_stage:
            #     close_gripper_action = np.array([0, 0, 0, 0, 0, 0, -force_value])  # Force to close the gripper
            #     self.step(close_gripper_action)
            #     if step % 10 == 0:
            #         self.render()
            #     step += 1
            

    def step(self, action):
        self.current_step += 1
        inc_contact_array = True if self.touch_rewards is not None else False
        next_state = self.panda_instance.step(np.array(action), 
                                              inc_contact_array=inc_contact_array,
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
        self.move_to_start_pos()
        initial_state = np.concatenate([self.panda_instance.data.qpos, self.panda_instance.data.qvel])
        if self.touch_rewards is not None:
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
        if self.destination_pos is not None:
            reward = -np.linalg.norm(self.panda_instance.data.xpos[self.panda_instance.body_name_to_idx["ee_ref_container"]] - self.destination_pos)
            # ??? nicely done copilot
            # https://www.youtube.com/watch?v=3KquFZYi6L0
            if self.center_rewards:
                reward -= self.starting_reward
            return reward
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
        if self.center_rewards:
            reward -= self.starting_reward # TODO: need to check that this is correct when we're using potential shaping
        if self.use_potential_shaping:
            reward += shaped_reward - self.last_shaped_reward
            self.last_shaped_reward = shaped_reward
        else:
            reward += shaped_reward
        return reward

    def is_done(self):
        return self.current_step >= self.max_steps



args = parse_args()
# Example: EVERYTHING MUST TOUCH EVERYTHING
# touch_rewards = np.ones(21)
# shaping_rewards = np.ones(28)/100
# shaping_rewards[21:] / 100
touch_rewards = np.ones(15) / 10 if args.use_touch_rewards else None
shaping_rewards = np.ones(21) / 1000 if args.use_shaping_rewards else None
if args.destination_pos.lower() != 'none':
    destination_pos = np.array([float(x) for x in args.destination_pos.split(',')])
else:
    destination_pos = None

env = PandaEnv(touch_rewards, shaping_rewards, destination_pos, max_steps=args.max_steps, use_potential_shaping=False, center_rewards=args.center_rewards, start_pos = args.starting_position)

# first things first, wandb
run = wandb.init(project="panda-mujoco", sync_tensorboard=True)

# as mentioned below, use a random number just to keep things distinct
# actually, we'll go ahead and use directories for this:
# runs/{random_number}/*
# random_number = np.random.randint(100000)
# actually that's dumb, we should use the date/time instead
# time = datetime.datetime.now()
# and actually let's add a custom message as well
# get a message from the user:
# message = input("Enter a name for this run: ")
# name = f"{message} {time}"

# change of plans; didn't like all of that. Let's use the wandb run name instead:
name = run.name


# first make runs/{name}, also making runs if it doesn't exist
os.makedirs(f"runs/{name}", exist_ok=True)
# then make runs/{name}/models and runs/{name}/images
os.makedirs(f"runs/{name}/models", exist_ok=True)
os.makedirs(f"runs/{name}/images", exist_ok=True)

if args.algorithm == 'PPO':
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{name}/tensorboard/", gamma=args.gamma)
elif args.algorithm == 'SAC':
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{name}/tensorboard/", gamma=args.gamma)
elif args.algorithm == 'TQC':
    # Assuming TQC is supported by stable_baselines3 and uses a similar interface
    model = TQC("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{name}/tensorboard/", gamma=args.gamma)
# or SAC instead:
# model = SAC(
#     "MlpPolicy", 
#     env, 
#     verbose=1, 
#     tensorboard_log=f"runs/{name}/tensorboard/", 
#     gamma=1,
# )
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

model.learn(total_timesteps=args.learning_steps, callback=wandb_callback)
# model.learn(total_timesteps=10000, callback=wandb_callback)

obs, _ = env.reset()
for i in range(args.test_steps):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    if i % args.test_render_interval == 0:
        # print(action)
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
# model = PPO.load(f"runs/{name}/models/panda_model")