from panda import Panda
import numpy as np
from PIL import Image
import mujoco
import mujoco.viewer
import time

panda_instance = Panda()

# This part is supposed to test the image generating stuff
for i in range(10000):
    panda_instance.step(np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]))

    if i % 1000 == 0:
        print(panda_instance.make_contact_array().tolist())


img_as_array = panda_instance.render()

Image.fromarray(img_as_array).save("test.png")