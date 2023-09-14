from panda import Panda
import numpy as np
from PIL import Image
import mujoco

panda_instance = Panda()

for i in range(10000):
    panda_instance.step(np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]))
    panda_instance.make_contact_array()

img_as_array = panda_instance.render()

Image.fromarray(img_as_array).save("test.png")

pass
