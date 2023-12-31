import os

# os.environ["MUJOCO_GL"] = "egl"
# os.environ["MUJOCO_GL"] = "glfw"
os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import numpy as np

module_root = os.path.dirname(__file__)
panda_xml_path = os.path.join(module_root, "assets/franka_emika_panda/scene.xml")


class Panda:
    @classmethod
    def init_static_(cls):
        cls.model = mujoco.MjModel.from_xml_path(panda_xml_path)
        cls.item_names = [
            "red_cube",
            "green_long",
            "yellow_flat",
            "purple_ball",
            "orange_cylinder",
            "blue_sponge",
            # "brown_towel_B4_4",
        ]

        # Get the names of the bodies (surprisingly hard)
        body_names = []
        for start_idx in cls.model.name_bodyadr:
            this_string_unterminated = cls.model.names[start_idx:].decode("ascii")
            this_string = this_string_unterminated.split("\x00", 1)[0]
            body_names.append(this_string)

        # Get the body ids of the items
        cls.item_body_ids = np.array(
            [body_names.index(name) for name in cls.item_names]
        )

        # WILL'S MODIFICATION: just doing what chatgpt told me to
        cls.body_name_to_idx = {name: idx for idx, name in enumerate(body_names)}

        # Get the 1d indices of the upper triangle of the contact matrix on a flattened matrix
        n_items = len(cls.item_names)
        cls.triu_indices = np.stack(
            np.triu_indices(n_items, k=1),
            1,
        ) @ np.array([n_items, 1])

        # What is the numbering of this element of the upper triangular in the whole matrix
        cls.rev_triu_indices = np.empty(cls.triu_indices.max() + 1, dtype=np.int32)
        cls.rev_triu_indices[:] = -1
        cls.rev_triu_indices[cls.triu_indices] = np.arange(len(cls.triu_indices))

        # Make an array which maps from geom_id to index in item array (item_id)
        cls.geom_id_to_item_id = np.empty_like(cls.model.geom_bodyid, dtype=np.int32)
        cls.geom_id_to_item_id[:] = -1
        for item_id, body_id in np.ndenumerate(cls.item_body_ids):
            cls.geom_id_to_item_id[cls.model.geom_bodyid == body_id] = item_id

    def __init__(self):
        self.renderer = None
        self.data = None
        self.reset()

    def reset(self, randomize=False):
        self.data = mujoco.MjData(self.model)

        if randomize:
            constraints = {
                "red_cube": [0.2, 0.4, 0.1, 0.2, 0.05, 0.1],
                "green_long": [0.2, 0.4, -0.2, -0.1, 0.05, 0.1],
                "yellow_flat": [0.35, 0.45, -0.1, 0.1, 0.05, 0.1],
                "purple_ball": [0.2, 0.3, 0.15, 0.25, 0.05, 0.1],
                "orange_cylinder": [0.2, 0.3, -0.25, -0.15, 0.05, 0.1],
                "blue_sponge": [0.45, 0.55, 0.15, 0.25, 0.1, 0.2],
                "brown_towel_B4_4": [0.65, 0.75, -0.1, 0.1, 0.1, 0.2],
            }

            for object_name, (
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
            ) in constraints.items():
                idx = self.body_name_to_idx[object_name]
                random_pos = np.array(
                    [
                        np.random.uniform(x_min, x_max),
                        np.random.uniform(y_min, y_max),
                        np.random.uniform(z_min, z_max),
                    ]
                )
                # self.data.qpos[idx:idx+3] = random_pos  # Assuming the position is stored in qpos[idx:idx+3]
                try:
                    self.data.qpos[idx * 7 : idx * 7 + 3] = random_pos
                except:
                    assert (
                        object_name == "brown_towel_B4_4"
                    ), "Oops, it's not that blasted towel again!"

        self.data.qpos[0:9] = np.array(
            [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04]
        )
        # Update simulation
        mujoco.mj_forward(self.model, self.data)

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

    def step(self, action, inc_contact_array=False, inc_item_dists=False, inc_hand_dists=False):
        # Check the dims of the control and the action
        if action.shape != self.data.ctrl.shape:
            raise ValueError(
                f"Action shape {action.shape} does not match control shape {self.data.ctrl.shape}"
            )

        # self.data.ctrl[:6] += action[:6]
        # self.data.ctrl[6] = action[6]
        self.data.ctrl[:] = action
        mujoco.mj_step(Panda.model, self.data)
        ret = np.concatenate([self.data.qpos, self.data.qvel])
        # if include_distance_matrix:
        #     distance_vec = self.make_distance_matrix(return_vec=True)
        #     ret = np.concatenate([ret, distance_vec])
        if inc_contact_array:
            contact_array = self.make_contact_array()
            ret = np.concatenate([ret, contact_array])
        if inc_item_dists:
            item_dists = self.make_item_distance_vector()
            ret = np.concatenate([ret, item_dists])
        if inc_hand_dists:
            hand_dists = self.make_hand_distance_vector()
            ret = np.concatenate([ret, hand_dists])
        return ret

    def make_contact_array(self):
        # Get the contacts relevant to the items
        geom1 = self.data.contact.geom1
        geom2 = self.data.contact.geom2

        # Stack the contact info into one tensor so we only have to index once
        contact_bodies = np.stack([geom1, geom2], axis=-1)

        # Convert the geom ids to item ids
        contact_items = self.geom_id_to_item_id[contact_bodies]

        # Toss out contacts that are not between two items
        relevant_geoms = contact_items != -1
        contact_is_relevant = relevant_geoms[:, 0] & relevant_geoms[:, 1]
        relevant_contacts = contact_items[contact_is_relevant]

        # Sort the contacts so that it follows the upper triangular
        relevant_contacts = np.sort(relevant_contacts, axis=-1)

        # Convert to the 1d indices in a matrix (row major should make it upper triangular)
        relevant_contact_1d_indices = (
            relevant_contacts[:, 0] * len(self.item_body_ids) + relevant_contacts[:, 1]
        )

        # Convert to the 1d indices in the upper triangular 1d index list
        contact_pair_indices = self.rev_triu_indices[relevant_contact_1d_indices]

        # Make a one hot matrix of the contact pairs
        one_hots = np.eye(len(self.triu_indices))[contact_pair_indices]

        # And these all together to get rid of redundant contacts and add different contacts together
        boolean_result = one_hots.any(axis=0)

        # Return it as an int since booleans have cooties
        return boolean_result.astype(np.int32)

    def make_item_distance_vector(self):
        # Get xyz position of each object
        item_pos = self.data.xpos[self.item_body_ids]

        matrix = np.tile(item_pos[None], item_pos.shape[0])
        diffs = matrix - matrix.transpose(1, 0, 2)
        norm_matrix = np.linalg.norm(diffs, axis=-1)

        norm_vector = norm_matrix.flatten()[self.triu_indices]
        return norm_vector

    def make_hand_distance_vector(self, pairwise=False):
        # Get xyz position of each object
        item_pos = self.data.xpos[self.item_body_ids]

        # Get xyz position of the hand
        hand_pos = self.data.xpos[self.body_name_to_idx["ee_ref_container"]]

        # Get hand to item diffs
        hand_dists = item_pos - hand_pos

        # Get norm squared
        # hand_norms_sq = np.einsum("id,id->i", hand_dists, hand_dists)
        hand_norms_sq = np.linalg.norm(hand_dists, axis=-1) ** 2
        if not pairwise:
            return hand_norms_sq

        repeat_matrix = np.tile(hand_norms_sq[None], hand_norms_sq.shape[0])
        hand_norms_matrix = repeat_matrix + repeat_matrix.T

        hand_norms_vector = hand_norms_matrix.flatten()[self.triu_indices]

        return hand_norms_vector

    def make_reward_space(self):
        hand_dists = self.make_hand_distance_vector(pairwise=True)
        item_dists = self.make_item_distance_vector()
        contact_vector = self.make_contact_array()

        # You might have to scale these, or otherwise modify the scaling with a log or something
        return contact_vector - hand_dists - item_dists

    def compute_hand_to_point_reward(self, target_pos):
        """
        Computes a reward based on the distance of the robot hand to a target position.

        Args:
            target_pos (np.ndarray): A 3-element array representing the target (x, y, z) position.

        Returns:
            float: The computed reward.
        """
        # Get the current position of the robot hand
        hand_pos = self.data.xpos[self.body_name_to_idx["ee_ref_container"]]

        # Calculate the Euclidean distance between the hand and the target position
        distance = np.linalg.norm(hand_pos - target_pos)

        # Compute the reward. This can be a simple inverse of the distance, or any other function
        # that appropriately represents the desired reward behavior. Here, we use an exponential
        # decay function as an example.
        reward = -distance

        return reward


Panda.init_static_()