from panda import Panda
import numpy as np
from PIL import Image
import mujoco
import mujoco.viewer
import time
import os

panda_instance = Panda()
panda_instance.reset()

# m = panda_instance.model
# d = panda_instance.data

# Step size for moving the arm, adjust as needed
step_size = 1


# Reset environment and hand position
panda_instance.reset()

initial_state = np.concatenate([panda_instance.data.qpos, panda_instance.data.qvel])
# print size of state
# print(f"Size of state: {initial_state.shape}")

# Optionally, render and save the image
# img_as_array = panda_instance.render()
# if not os.path.exists("test_qpos_images"):
#     os.makedirs("test_qpos_images")
# Image.fromarray(img_as_array).save(f"test_qpos_images/before.png")

# print("Movement towards target completed.")

# let's try just doing the red cube for now. First, get its xpos:
# object_name = np.random.choice(Panda.item_names)
for object_name in Panda.item_names:
    object_idx = panda_instance.body_name_to_idx[object_name]
    print(f"Target Object: {object_name}")
    object_pos = panda_instance.data.xpos[object_idx]
    hand_pos = panda_instance.data.xpos[panda_instance.body_name_to_idx["ee_ref_container"]]
    step = 0
    max_steps_per_stage = 1000
    position_threshold = 0.1
    while np.linalg.norm(panda_instance.data.xpos - object_pos) > position_threshold and step < max_steps_per_stage:
        # Apply forces to move towards object_pos
        # self.step(move_down_action)
        direction_to_target = object_pos - hand_pos
        direction_to_target_normalized = direction_to_target / np.linalg.norm(direction_to_target)
        action = np.zeros(7)  # 7 control dimensions, adjust if different
        action[3:6] = direction_to_target_normalized * step_size # Apply movement in the 4th, 5th, 6th actuators
        action[6] = 0 # keep the gripper open
        panda_instance.step(action)
        if step % 10 == 0:
            # self.render()
            img = panda_instance.render()
            # save to "object_start/{object_name}/{stage}/{step}.png"
            # first make the directory
            os.makedirs(f"object_start/{object_name}", exist_ok=True)
            # then save the image
            Image.fromarray(img).save(f"object_start/{object_name}/{step}.png")
            # diagnostic: print out positions of hand and object
            print(f"Hand position: {hand_pos}")
            print(f"Object position: {object_pos}")
            # also print out the finger positions:
            # (their body names are left_finger and right_finger)
            left_finger_pos = panda_instance.data.xpos[panda_instance.body_name_to_idx["left_finger"]]
            right_finger_pos = panda_instance.data.xpos[panda_instance.body_name_to_idx["right_finger"]]
            print(f"Left finger position: {left_finger_pos}")
            print(f"Right finger position: {right_finger_pos}")
        step += 1
    # at the end, save the qpos (we'll just init qvel to 0)
    qpos = panda_instance.data.qpos
    # save to the same object_starts folder
    np.save(f"object_start/{object_name}/qpos.npy", qpos)

    # now close the gripper: (actually this doesn't seem to work)
    # maybe there's some bias in the robot, like elasticity
    # could fix this later by just moving the fingers to the object
    # while the gripper closes
    # whatevs
    # action = np.zeros(7)
    # for step in range(100):
    #     panda_instance.step(action)
    #     if step % 10 == 0:
    #         # self.render()
    #         img = panda_instance.render()
    #         # save to "object_start/{object_name}/{stage}/{step}.png"
    #         # first make the directory
    #         os.makedirs(f"object_start/{object_name}", exist_ok=True)
    #         # then save the image
    #         Image.fromarray(img).save(f"object_start/{object_name}/close_{step}.png")
    #         # diagnostic: print out positions of hand and object
    #         print(f"Closing hand position: {hand_pos}")
    #         print(f"Closing object position: {object_pos}")
    #         # also print out the finger positions:
    #         # (their body names are left_finger and right_finger)
    #         left_finger_pos = panda_instance.data.xpos[panda_instance.body_name_to_idx["left_finger"]]
    #         right_finger_pos = panda_instance.data.xpos[panda_instance.body_name_to_idx["right_finger"]]
    #         print(f"Closing left finger position: {left_finger_pos}")
    #         print(f"Closing right finger position: {right_finger_pos}")

        # step += 1