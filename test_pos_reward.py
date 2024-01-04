from panda import Panda
import numpy as np
from PIL import Image
import mujoco
import mujoco.viewer
import time
import os

panda_instance = Panda()
panda_instance.reset()

m = panda_instance.model
# d = panda_instance.data

# Define target position
target_position = np.array([0.5, -0.5, -0.5])  # Adjust this to your desired target position
target_position *= 10
# Number of steps to move towards the target
num_steps = 50  # Adjust this as needed

# Step size for moving the arm, adjust as needed
step_size = 10


# Reset environment and hand position
panda_instance.reset()

initial_state = np.concatenate([panda_instance.data.qpos, panda_instance.data.qvel])
# print size of state
print(f"Size of state: {initial_state.shape}")

for step in range(num_steps):
    # Get current hand position
    current_hand_position = panda_instance.data.xpos[panda_instance.body_name_to_idx["ee_ref_container"]]

    # Calculate the direction vector towards the target
    direction_to_target = target_position - current_hand_position
    direction_to_target_normalized = direction_to_target / np.linalg.norm(direction_to_target)

    # Create an action that moves the hand in the direction of the target
    action = np.zeros(7)  # 7 control dimensions, adjust if different
    action[3:6] = direction_to_target_normalized * step_size  # Apply movement in the 4th, 5th, 6th actuators
    # TEST
    # action = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    # Apply the action and step the simulation
    print(f"Step {step}, Current hand position: {current_hand_position}")
    panda_instance.step(action)

    # Compute and print the reward
    reward = panda_instance.compute_hand_to_point_reward(target_position)
    # also print the action
    print(f"Step {step}, Action: {action}")
    print(f"Step {step}, Reward: {reward}")

    # Optionally, render and save the image
    img_as_array = panda_instance.render(camera="wrist_cam")
    # put them in a folder called "test_pos_reward_images"
    # first make it
    if not os.path.exists("test_pos_reward_images"):
        os.makedirs("test_pos_reward_images")
    # Image.fromarray(img_as_array).save(f"step_{step}.png")
    Image.fromarray(img_as_array).save(f"test_pos_reward_images/step_{step}.png")

print("Movement towards target completed.")
