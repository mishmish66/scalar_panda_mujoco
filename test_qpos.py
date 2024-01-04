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

# Optionally, render and save the image
img_as_array = panda_instance.render()
if not os.path.exists("test_qpos_images"):
    os.makedirs("test_qpos_images")
Image.fromarray(img_as_array).save(f"test_qpos_images/before.png")

# now load the state from object_start/blue_sponge/qpos/final.npy
# and set the state
# load the state
state = np.load("object_start/blue_sponge/qpos/final.npy")
# # set the state
# panda_instance.data.qpos[:] = state
# panda_instance.data.qvel[:] = 0
# panda_instance.data.ctrl[:] = 0
# action = np.zeros(7)
# panda_instance.step(action)
# let's also try the set_state function
# panda_instance.sim.set_state(state)
# that didn't work
panda_instance.data.qpos[:] = state
# just for funsies
# panda_instance.data.qpos[panda_instance.model.get_joint_qpos_addr("panda_finger_joint1")] = 0.04
# hand_pos = self.data.xpos[self.body_name_to_idx["ee_ref_container"]]
# panda_instance.data.xpos[panda_instance.body_name_to_idx['ee_ref_container']] = np.array([0.5, -0.5, -0.5])
print("red before:" + str(panda_instance.data.qpos[panda_instance.body_name_to_idx['red_cube']]))
print("red xpos before:" + str(panda_instance.data.xpos[panda_instance.body_name_to_idx['red_cube']]))
print("ee_ref_container before:" + str(panda_instance.data.qpos[panda_instance.body_name_to_idx['ee_ref_container']]))
# panda_instance.data.qpos[panda_instance.body_name_to_idx['red_cube']] = panda_instance.data.qpos[panda_instance.body_name_to_idx['ee_ref_container']]
# panda_instance.data.qpos += 100 # ooh la la
# panda_instance.data.qpos[panda_instance.body_name_to_idx['red_cube']:panda_instance.body_name_to_idx['red_cube']+6] += 10000
panda_instance.data.joint("testred").qpos += 100
print("red after:" + str(panda_instance.data.qpos[panda_instance.body_name_to_idx['red_cube']:panda_instance.body_name_to_idx['red_cube']+6]))
print("ee_ref_container after:" + str(panda_instance.data.qpos[panda_instance.body_name_to_idx['ee_ref_container']]))
# ooh la la
# print(panda_instance.body_name_to_idx)
# {'world': 0, 'link0': 1, 'link1': 2, 'link2': 3, 'link3': 4, 'link4': 5, 'link5': 6, 'link6': 7, 'link7': 8, 'hand': 9, 'left_finger': 10, 'right_finger': 11, 'ee_ref_container': 12, 'red_cube': 13, 'green_long': 14, 'yellow_flat': 15, 'purple_ball': 16, 'orange_cylinder': 17, 'blue_sponge': 18, 'B0_0_0': 19, 'B0_0_1': 20, 'B0_0_2': 21, 'B0_0_3': 22, 'B0_1_0': 23, 'B0_1_1': 24, 'B0_1_2': 25, 'B0_1_3': 26, 'B0_2_0': 27, 'B0_2_1': 28, 'B0_2_2': 29, 'B0_2_3': 30, 'B0_3_0': 31, 'B0_3_1': 32, 'B0_3_2': 33, 'B0_3_3': 34, 'B1_0_0': 35, 'B1_0_1': 36, 'B1_0_2': 37, 'B1_0_3': 38, 'B1_1_0': 39, 'B1_1_3': 40, 'B1_2_0': 41, 'B1_2_3': 42, 'B1_3_0': 43, 'B1_3_1': 44, 'B1_3_2': 45, 'B1_3_3': 46, 'B2_0_0': 47, 'B2_0_1': 48, 'B2_0_2': 49, 'B2_0_3': 50, 'B2_1_0': 51, 'B2_1_3': 52, 'B2_2_0': 53, 'B2_2_3': 54, 'B2_3_0': 55, 'B2_3_1': 56, 'B2_3_2': 57, 'B2_3_3': 58, 'B3_0_0': 59, 'B3_0_1': 60, 'B3_0_2': 61, 'B3_0_3': 62, 'B3_1_0': 63, 'B3_1_1': 64, 'B3_1_2': 65, 'B3_1_3': 66, 'B3_2_0': 67, 'B3_2_1': 68, 'B3_2_2': 69, 'B3_2_3': 70, 'B3_3_0': 71, 'B3_3_1': 72, 'B3_3_2': 73, 'B3_3_3': 74}
print(len(panda_instance.data.qpos))
# print(panda_instance.data.joint("red_cube").qpos)
# print(panda_instance.data.joint("testred").qpos)
mujoco.mj_forward(panda_instance.model, panda_instance.data)
# mujoco.mj_step(panda_instance.model, panda_instance.data)
# print("red after forward:" + str(panda_instance.data.qpos[panda_instance.body_name_to_idx['red_cube']]))
print("red after forward:" + str(panda_instance.data.qpos[panda_instance.body_name_to_idx['red_cube']:panda_instance.body_name_to_idx['red_cube']+6]))
print("ee_ref_container after forward:" + str(panda_instance.data.qpos[panda_instance.body_name_to_idx['ee_ref_container']]))
print("red xpos after forward:" + str(panda_instance.data.xpos[panda_instance.body_name_to_idx['red_cube']]))
print("ee_ref_container xpos after forward:" + str(panda_instance.data.xpos[panda_instance.body_name_to_idx['ee_ref_container']]))
# mujoco.mj_step(panda_instance.model, panda_instance.data)
# actually now I'm going to test how the gripper actuator works
# panda_instance.data.ctrl[7] = 255
action = np.zeros(7)
action[6] = 255
for i in range(10):
    panda_instance.step(action)
    img_as_array = panda_instance.render()
    Image.fromarray(img_as_array).save(f"test_qpos_images/gripper_positive_{i}.png")
action[6] = 0
for i in range(10):
    panda_instance.step(action)
    img_as_array = panda_instance.render()
    Image.fromarray(img_as_array).save(f"test_qpos_images/gripper_zero_{i}.png")

# now render again to see if it worked
img_as_array = panda_instance.render()
# if not os.path.exists("test_qpos_images"):
#     os.makedirs("test_qpos_images")
Image.fromarray(img_as_array).save(f"test_qpos_images/after.png")

# print("Movement towards target completed.")
