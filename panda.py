import mujoco
import numpy as np

import os

module_root = os.path.dirname(__file__)
panda_xml_path = os.path.join(module_root, "assets/franka_emika_panda/scene.xml")


class Panda:
    @classmethod
    def init_static_(cls):
        cls.model = mujoco.MjModel.from_xml_path(panda_xml_path)
        item_names = [
            "red_cube",
            "green_long",
            "yellow_flat",
            "purple_ball",
            "orange_cylinder",
        ]

        # Get the names of the bodies (surprisingly hard)
        body_names = []
        for start_idx in cls.model.name_bodyadr:
            this_string_unterminated = cls.model.names[start_idx:].decode("ascii")
            this_string = this_string_unterminated.split("\x00", 1)[0]
            body_names.append(this_string)

        # Get the body ids of the items
        cls.item_body_ids = np.array([body_names.index(name) for name in item_names])

        # Get the 1d indices of the upper triangle of the contact matrix on a flattened matrix
        n_items = len(item_names)
        cls.triu_indices = np.stack(
            np.triu_indices(n_items, k=1),
            1,
        ) @ np.array([n_items, 1])

        # What is the numbering of this element of the upper triangular in the whole matrix
        cls.rev_triu_indices = np.empty(cls.triu_indices.max() + 1, dtype=np.int32)
        cls.rev_triu_indices[:] = -1
        cls.rev_triu_indices[cls.triu_indices] = np.arange(len(cls.triu_indices))

        # Give each item an ID
        cls.body_id_to_item_id = np.empty(len(body_names), dtype=np.int32)
        cls.body_id_to_item_id[:] = -1
        cls.body_id_to_item_id[cls.item_body_ids] = np.arange(len(cls.item_body_ids))

        # Which body ids are items
        cls.body_id_is_item = np.zeros(len(body_names), dtype=bool)
        cls.body_id_is_item[cls.item_body_ids] = True

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

        self.renderer.update_scene(self.data, camera)
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

    def make_contact_array(self):
        # Get the contacts relevant to the items
        geom1 = self.data.contact.geom1
        geom2 = self.data.contact.geom2
        
        # Convert geom ids to body ids
        body1 = self.model.geom_bodyid[geom1]
        body2 = self.model.geom_bodyid[geom2]

        # Stack the contact info into one tensor so we only have to index once
        contact_bodies = np.stack([body1, body2], axis=-1)

        # Check which contacts are really taking place between two items
        relevant_bodies = self.body_id_is_item[contact_bodies]
        contact_is_relevant = np.logical_and.reduce(relevant_bodies, axis=-1)
        
        # Pull out the relevant contacts
        relevant_contacts = contact_bodies[contact_is_relevant]
        
        # Convert body ids in relevant contacts to item ids (We can do this since we know both contacting parties are items)
        relevant_item_contacts = self.body_id_to_item_id[relevant_contacts]

        # Sort the contacts so that it follows the upper triangular
        relevant_item_contacts = np.sort(relevant_item_contacts, axis=-1)

        # Convert to the 1d indices in a matrix (row major) (sorting should make it upper triangular)
        relevant_contact_1d_indices = (
            relevant_item_contacts[:, 0] * len(self.item_body_ids)
            + relevant_item_contacts[:, 1]
        )

        # Convert to the 1d indices in the upper triangular 1d index list
        contact_pair_indices = self.rev_triu_indices[relevant_contact_1d_indices]

        # Make a one hot matrix of the contact pairs
        one_hots = np.eye(len(self.triu_indices))[contact_pair_indices]

        # And these all together to get rid of redundant contacts and add different contacts together
        boolean_result = np.logical_and.reduce(one_hots, axis=0)
        
        # Return it as an int since booleans have cooties
        return boolean_result.astype(np.int32)


Panda.init_static_()
