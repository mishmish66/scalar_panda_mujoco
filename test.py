from panda import Panda
import numpy as np
from PIL import Image
import mujoco
import mujoco.viewer
import time

panda_instance = Panda()

# This part is supposed to test the image generating stuff
# for i in range(10000):
#     panda_instance.step(np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]))

#     if i % 1000 == 0:
#         print(panda_instance.make_contact_array().tolist())


# img_as_array = panda_instance.render()

# Image.fromarray(img_as_array).save("test.png")

m = panda_instance.model
d = panda_instance.data

with mujoco.viewer.launch_passive(m, d) as viewer:
    
    start = time.time()
    
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        
        panda_instance.step(np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]))
        panda_instance.make_distance_matrix()

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)