import os
import json
import glob
from PIL import Image
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm as tqdm


def build_dataset(
    data_path,
    time_sequence_length=6,
    predicting_next_ts=True,
    num_train_episode=200,
    num_val_episode=100,
    cam_view=["front"],
    language_embedding_size=512,
):
    """
    This function is for building the training and validation dataset

    Parameters:
    - data_path(str): locates the path where the dataset is stored
            the dataset path should have the following file structures:
                - [robotname]_[taskname]
                    - [cam_view_0]
                        - data_000
                            - rgb # where those image stored
                                - image_001.png
                                - image_002.png
                                - ...
                            - results.csv # robot actions stored
                            - results_raw.csv # joint and target object position stored
                        - data_001
                        - ...
                    - [cam_view_1]
                        - data_000
                        - data_001
                        - ...
                    - ...
    - time_sequence_length(int) : number of history length input for RT-1 model,
        6 means current frame image and past 5 frames of images will be packed and input to RT-1
    - predicting_next_ts(bool) : in our dataset's results.csv and results_raw.csv, we stored current frame's action and joint status.
        if we want to predict next frame's action, this option needs to be True and result in the 1 step offset reading on csv files
        this differs between the samplings method of different dataset.
    - num_train_episode(int) : specifies numbers of training episodes
    - num_train_episode(int) : specifies numbers of validation episodes
    - cam_view(list of strs) : camera views used for training.

    Returns:
    - train_dataset(torch.utils.data.Dataset)
    - val_dataset(torch.utils.data.Dataset)
    """

    with open(os.path.join(data_path, cam_view[0], "dataset_info.json"), "r") as f:
        info = json.load(f)
    episode_length = info["episode_length"]
    episode_dirs = sorted(glob.glob(data_path + "/" + cam_view[0] + "/*/"))
    assert len(episode_dirs) == len(
        episode_length
    ), "length of episode directories and episode length not equal, check dataset's dataset_info.json"
    perm_indice = torch.randperm(len(episode_dirs)).tolist()
    dirs_lengths = dict(
        episode_dirs=np.array(episode_dirs)[perm_indice],
        episode_length=np.array(episode_length)[perm_indice],
    )
    train_episode_dirs = dirs_lengths["episode_dirs"][:num_train_episode]
    train_episode_length = dirs_lengths["episode_length"][:num_train_episode]
    val_episode_dirs = dirs_lengths["episode_dirs"][
        num_train_episode : num_train_episode + num_val_episode
    ]
    val_episode_length = dirs_lengths["episode_length"][
        num_train_episode : num_train_episode + num_val_episode
    ]

    train_dataset = IODataset(
        episode_dirs=train_episode_dirs,
        episode_length=train_episode_length,
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_view=cam_view,
        language_embedding_size=language_embedding_size,
    )
    val_dataset = IODataset(
        episode_dirs=val_episode_dirs,
        episode_length=val_episode_length,
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_view=cam_view,
        language_embedding_size=language_embedding_size,
    )
    return train_dataset, val_dataset


class IODataset(Dataset):
    def __init__(
        self,
        episode_dirs,
        episode_length,
        time_sequence_length=6,
        predicting_next_ts=True,
        cam_view=["front"],
        robot_dof=9,
        language_embedding_size=512,
    ):
        self._cam_view = cam_view
        self.predicting_next_ts = predicting_next_ts
        self._time_sequence_length = time_sequence_length
        self._episode_length = episode_length
        self.querys = self.generate_history_steps(episode_length)
        self._episode_dirs = episode_dirs
        self.keys_image = self.generate_fn_lists(self._episode_dirs)
        self.values, self.num_zero_history_list = self.organize_file_names()
        self._robot_dof = robot_dof
        self._language_embedding_size = language_embedding_size

    def generate_fn_lists(self, episode_dirs):
        """
        This function globs all the image path in the dataset
        Parameters:
        - episode_dirs(list of strs): directories where image is stored, etc:
            - [robotname]_[taskname]
                - [cam_view_0]
                    - data_000
                    - data_001
                    - data_002
                    - ...
        Returns:
        - keys(list of strs): all globbed image filename in a list
        """
        keys = []
        for ed in episode_dirs:
            image_files = sorted(glob.glob(f"{ed}rgb/*.png"))
            keys.append(image_files)
        return keys

    def generate_history_steps(self, episode_length):
        """
        This function generates the step for current frame and history frames
        Parameters:
        - episode_length(list of int): number of episode lengths for each episode
        Returns:
        - keys(list of tensors): history steps for each data
        """
        querys = []
        for el in episode_length:
            q = torch.cat(
                (
                    [
                        torch.arange(el)[:, None] - i
                        for i in range(self._time_sequence_length)
                    ]
                ),
                dim=1,
            )
            q[q < 0] = -1
            querys.append(q.flip(1))
        return querys

    def organize_file_names(self):
        """
        This function generates the infor for each data, including how many zeros were padded
        data's episode directory, image filenames, and all the other parameters for data
        Parameters:
        -
        Returns:
        - values(list): each value including
            - num_zero_history: when we read at initial frames of a episode, it doesn't have history,
                then we need to pad zeros to make sure these aligns to data with history frames.
                this number specified how many frames of zero is padded
            - episode_dir: the episode directory where this data is stored
            - img_fns = img_fns: the images this data should read
            - query_index = index of this data in this episode
            - episode_length = total length of this episode
        """
        values = []
        num_zero_history_list = []
        for i, (query, key_img, ed) in enumerate(
            zip(self.querys, self.keys_image, self._episode_dirs)
        ):
            for q in query:
                img_fns = []
                for img_idx in q:
                    img_fns.append(key_img[img_idx] if img_idx >= 0 else None)
                num_zero_history = (q < 0).sum()
                num_zero_history_list.append(int(num_zero_history))
                values.append(
                    dict(
                        num_zero_history=num_zero_history,
                        episode_dir=ed,
                        img_fns=img_fns,
                        query_index=q,
                        episode_length=self._episode_length[i],
                    )
                )
        return values, num_zero_history_list

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        value = self.values[idx]
        img = self.get_image(value["img_fns"])
        lang = self.get_language_instruction()
        ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint = self.get_ee_data(
            value["episode_dir"], value["query_index"], value["num_zero_history"]
        )
        terminate_episode = self.get_episode_status(
            value["episode_length"], value["query_index"], value["num_zero_history"]
        )
        sample_obs = {
            "image": img.float().permute(0, 3, 1, 2),
            # we permute the channel dimension to the second dimension to cope with rt1's convolution layers
            "natural_language_embedding": torch.Tensor(lang).long(),
            "joint_position": torch.Tensor(joint).float(),
            # "tar_obj_pose": torch.tensor(tar_obj_pose).float(),
        }
        sample_action = {
            "world_vector": torch.Tensor(ee_pos_cmd).float(),
            "rotation_delta": torch.Tensor(ee_rot_cmd).float(),
            "gripper_closedness_action": torch.Tensor(gripper_cmd).float(),
            "terminate_episode": torch.Tensor(terminate_episode.argmax(-1)).long(),
        }

        return sample_obs, sample_action

    def get_image(self, img_fns):
        """
        This function generates the step for current frame and history frames
        Parameters:
        - episode_length(list of int): number of episode lengths for each episode
        Returns:
        - keys(list of tensors): history steps for each data
        """
        imgs = []
        for img_fn in img_fns:
            img_multi_view = []
            for c_v in self._cam_view:
                img_multi_view.append(
                    np.array(Image.open(img_fn.replace(self._cam_view[0], c_v)))
                    if img_fn != None
                    else np.zeros_like(Image.open(img_fns[-1]))
                )
            img = np.concatenate(img_multi_view, axis=0)
            imgs.append(torch.from_numpy(img[:, :, :3]))
        return torch.stack(imgs, dim=0) / 255.0

    def get_ee_data(self, episode_dir, query_index, pad_step_num):
        """
        This function reads the csvs for ground truth robot actions, robot joint status and target object position and orientation:
        Parameters:
        - episode_dir(str): directory where the results.csv and results_raw.csv is stored
        - query_index(tensor): index where exact data is read, padded zeros has a special index of -1
        - pad_step_num(int): how many timestep of zeros is padded
        Returns:
        - ee_pos_cmd(np.array): stores the ground truth command for robot move in position(x, y, z)
        - ee_rot_cmd(np.array): stores the ground truth command for robot move in rotation(rx, ry, rz)
        - gripper_cmd(np.array): stores the ground truth command for robot's gripper open or close
        - joint(np.array): stores the robot's joint status, which can be used to calculate ee's position
        - tar_obj_pose: stores the target object's position and orientation (x, y, z, rx, ry, rz)
        """
        start_idx = query_index[(query_index > -1).nonzero()[0, 0]]
        end_idx = query_index[-1]
        visual_data_filename = f"{episode_dir}result.csv"
        raw_data = pd.read_csv(visual_data_filename)
        visual_data_filename_raw = f"{episode_dir}result_raw.csv"
        raw_raw_data = pd.read_csv(visual_data_filename_raw)
        if self.predicting_next_ts:
            """
            if predicting next timestep's results, then we shift first column to last column
            """
            first_row = raw_data.iloc[0]
            raw_data = raw_data.iloc[1:]
            raw_data = pd.concat([raw_data, first_row.to_frame().T], ignore_index=True)
            first_row = raw_raw_data.iloc[0]
            raw_raw_data = raw_raw_data.iloc[1:]
            raw_raw_data = pd.concat(
                [raw_raw_data, first_row.to_frame().T], ignore_index=True
            )
        # position has 3 dimensions [x, y, z]
        ee_pos_cmd = np.zeros([pad_step_num, 3])
        # rotation has 3 dimensions [rx, ry, rz]
        ee_rot_cmd = np.zeros([pad_step_num, 3])
        # gripper has 1 dimension which controls open/close of the gripper
        gripper_cmd = np.zeros([pad_step_num, 1])
        # we are using Franka Panda robot, whose has 9 dofs of joint
        joint = np.zeros([pad_step_num, 9])
        # tar_obj_pose is 7 dimension [x,y,z,rx,ry,rz,w]
        # however, in this version we are not using tar_obj_pose
        tar_obj_pose = np.zeros([pad_step_num, 7])
        ee_pos_cmd = np.vstack(
            (
                ee_pos_cmd,
                raw_data.loc[
                    start_idx:end_idx,
                    [f"ee_command_position_{ax}" for ax in ["x", "y", "z"]],
                ].to_numpy(),
            )
        )
        ee_rot_cmd = np.vstack(
            (
                ee_rot_cmd,
                raw_data.loc[
                    start_idx:end_idx,
                    [f"ee_command_rotation_{ax}" for ax in ["x", "y", "z"]],
                ].to_numpy(),
            )
        )
        joint = np.vstack(
            (
                joint,
                raw_raw_data.loc[
                    start_idx:end_idx,
                    [f"joint_{str(ax)}" for ax in range(self._robot_dof)],
                ].to_numpy(),
            )
        ).astype(np.float32)
        # tar_obj_pose = np.vstack(
        #     (
        #         tar_obj_pose,
        #         raw_raw_data.loc[
        #             start_idx:end_idx,
        #             [
        #                 f"tar_obj_pose_{ax}"
        #                 for ax in ["x", "y", "z", "rx", "ry", "rz", "rw"]
        #             ],
        #         ].to_numpy(),
        #     )
        # )
        gripper_data = (
            raw_data.loc[start_idx:end_idx, "gripper_closedness_commanded"]
            .to_numpy()
            .reshape(-1, 1)
        )
        gripper_cmd = np.vstack((gripper_cmd, gripper_data))
        return ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint

    def get_language_instruction(self):
        """
        since we are only training single-task model, this language embedding is set as constant.
        modify it to language instructions if multi-task model is training.
        it seems that google directly loads embedded language instruction from its language model
        this results in our loading a language embedding instead of language sentence
        """
        # lan = value["ddr"].split("/")[-1]
        # lan = self.classes[re.match(r"^\d{8}_(\w+)_\d+_\d+$", lan).group(1)]
        return torch.ones(self._time_sequence_length) * 3

    def get_episode_status(self, episode_length, query_index, pad_step_num):
        """
        This function is to find whether current frame and history frame is start or middle or end of the episode:
        Parameters:
        - episode_length(int): length of current episode
        - query_index(tensor): index where exact data is read, padded zeros has a special index of -1
        - pad_step_num(int): how many timestep of zeros is padded
        Returns:
        - episode_status(np.array): specifies status(start, middle or end) of each frame in history
        """
        start_idx = query_index[(query_index > -1).nonzero()[0, 0]]
        end_idx = query_index[-1]
        episode_status = np.zeros([pad_step_num, 4], dtype=np.int32)
        episode_status[:, -1] = 1
        for i in range(start_idx, end_idx + 1):
            status = np.array(
                [i == 0, i not in [0, episode_length - 2], i == episode_length - 2, 0],
                dtype=np.int32,
            )
            episode_status = np.vstack((episode_status, status))
        if pad_step_num > 0:
            episode_status[pad_step_num] = np.array([1, 0, 0, 0])
        return episode_status


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def load_config_from_json(json_path):
        with open(json_path, "r") as f:
            config = json.load(f)
        return config

    args = load_config_from_json("train_config.json")
    train_dataset, val_dataset = build_dataset(
        data_path=args["data_path"],
        time_sequence_length=args["time_sequence_length"],
        predicting_next_ts=args["predicting_next_ts"],
        num_train_episode=100,
        num_val_episode=args["num_val_episode"],
        cam_view=args["cam_view"],
        language_embedding_size=args["network_configs"]["language_embedding_size"],
    )
    dataloader = DataLoader(train_dataset, 32, num_workers=32, shuffle=False)
    posx = []
    posy = []
    posz = []
    rotx = []
    roty = []
    rotz = []
    for b in tqdm(dataloader):
        _, action = b
        posx.append(float(action["world_vector"][0][0][0]))
        posy.append(float(action["world_vector"][0][0][1]))
        posz.append(float(action["world_vector"][0][0][2]))
        rotx.append(float(action["rotation_delta"][0][0][0]))
        roty.append(float(action["rotation_delta"][0][0][1]))
        rotz.append(float(action["rotation_delta"][0][0][2]))
    plt.subplot(2, 3, 1)
    plt.hist(posx, bins=256, label="posx")
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.hist(posy, bins=256, label="posy")
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.hist(posz, bins=256, label="posz")
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.hist(rotx, bins=256, label="rotx")
    plt.legend()
    plt.subplot(2, 3, 5)
    plt.hist(roty, bins=256, label="roty")
    plt.legend()
    plt.subplot(2, 3, 6)
    plt.hist(rotz, bins=256, label="rotz")
    plt.legend()
    plt.show()
