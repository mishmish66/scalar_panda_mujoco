import mujoco
import numpy as np

import os

module_root = os.path.dirname(__file__)
panda_xml_path = os.path.join(module_root, "assets/franka_emika_panda/scene.xml")


class Panda:
    model = mujoco.MjModel.from_xml_path(panda_xml_path)

    def __init__(self):
        self.renderer = None
        self.data = None
        self.reset()

    def reset(self):
        self.data = mujoco.MjData(self.model)

    def render(self, camera="topdown_cam", width=1024, height=1024):
        """Generates a render from the camera.

        Args:
            camera (str, optional): String name of camera. Either "topdown_cam" or "wrist_cam" Defaults to "topdown_cam".
            width (int, optional): Width of image to output. Defaults to 1024.
            height (int, optional): Height of image to output. Defaults to 1024.

        Returns:
            numpy.ndarray : An array of shape (height, width, 3) representing the RGB image.
        """

        # Reset the context if it is not big enough or it doesn't exist
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width, height)

        self.renderer.update_scene(self.data, "topdown_cam")
        img = self.renderer.render()
        return img

    def step(self, action):
        # Check the dims of the control and the action
        if action.shape != self.data.ctrl.shape:
            raise ValueError(
                f"Action shape {action.shape} does not match control shape {self.data.ctrl.shape}"
            )
        self.data.ctrl[:] = action
        mujoco.mj_step(Panda.model, self.data)
        
        return np.concatenate([self.data.qpos, self.data.qvel])
