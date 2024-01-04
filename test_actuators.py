from panda import Panda
import numpy as np
from PIL import Image
import mujoco
import mujoco.viewer
import time

panda_instance = Panda()
panda_instance.reset()

m = panda_instance.model
d = panda_instance.data

# Number of actuators, images per actuator, and steps between renderings
# num_actuators = len(panda_instance.model.actuator_length)
num_actuators = 7
num_images_per_actuator = 10
steps_between_renderings = 5  # Adjust this as needed

# Force to apply
force_value = 1.0  # Adjust as needed

for actuator_index in range(num_actuators):
    panda_instance.reset()
    # Apply force to each actuator one by one
    for image_index in range(num_images_per_actuator):
        # Apply force and step the simulation
        action = np.zeros(num_actuators)
        action[actuator_index] = force_value

        for _ in range(steps_between_renderings):
            panda_instance.step(action)
            action = np.zeros(num_actuators)  # Reset action after initial force application

        # Render and save the image
        img_as_array = panda_instance.render()
        Image.fromarray(img_as_array).save(f"actuator_{actuator_index}_step_{image_index}.png")

print("Actuator testing completed.")