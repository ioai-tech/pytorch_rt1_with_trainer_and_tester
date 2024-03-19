import os
import json
import glob
from PIL import Image
import pandas as pd
import numpy as np
import math
import copy
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import cstl
import pybullet as p
import pybullet_data as pdata


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def build_dataset(
    data_path,
    time_sequence_length,
    predicting_next_ts,
    cam_view,
    language_embedding_size,
    H,
    future_traj_horizon,
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

    # with open(os.path.join(data_path, cam_view[0], "dataset_info.json"), "r") as f:
    #     info = json.load(f)
    # episode_length = info["episode_length"]
    # episode_dirs = sorted(glob.glob(data_path + "/" + cam_view[0] + "/*/"))
    # num_train_episode = len(episode_dirs) - 50
    # num_val_episode = 50
    # assert len(episode_dirs) == len(
    #     episode_length
    # ), "length of episode directories and episode length not equal, check dataset's dataset_info.json"
    # perm_indice = torch.randperm(len(episode_dirs)).tolist()
    # dirs_lengths = dict(
    #     episode_dirs=np.array(episode_dirs)[perm_indice],
    #     episode_length=np.array(episode_length)[perm_indice],
    # )
    # train_episode_dirs = dirs_lengths["episode_dirs"][:num_train_episode]
    # train_episode_length = dirs_lengths["episode_length"][:num_train_episode]
    # val_episode_dirs = dirs_lengths["episode_dirs"][
    #     num_train_episode : num_train_episode + num_val_episode
    # ]
    # val_episode_length = dirs_lengths["episode_length"][
    #     num_train_episode : num_train_episode + num_val_episode
    # ]

    train_dataset = IODataset(
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_view=cam_view,
        language_embedding_size=language_embedding_size,
        H=H,
        future_traj_horizon=future_traj_horizon,
    )
    # val_dataset = IODataset(
    #     episode_dirs=val_episode_dirs,
    #     episode_length=val_episode_length,
    #     time_sequence_length=time_sequence_length,
    #     predicting_next_ts=predicting_next_ts,
    #     cam_view=cam_view,
    #     language_embedding_size=language_embedding_size,
    #     H=H,
    #     future_traj_horizon=future_traj_horizon,
    # )
    return train_dataset, None


class IODataset(Dataset):
    def __init__(
        self,
        episode_dir,
        time_sequence_length=6,
        predicting_next_ts=True,
        cam_view=["front"],
        robot_dof=9,
        language_embedding_size=512,
        H=[20, 60],
        future_traj_horizon=21,
    ):
        self._cam_view = cam_view
        self.data = pd.read_hdf(
            "/home/io011/nfs/factorworld_dataset/version_pick_only_fa_/Panda/data.h5",
            key="data",
        )
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pdata.getDataPath())
        self.panda = p.loadURDF(
            "franka_panda/panda.urdf",
            [0, -0.1, 0.62],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        self.panda_ee_index = 11

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
        for ed in tqdm(episode_dirs, desc="keys.append(image_files)"):
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

    def generate_future_traj_step(self, episode_length):
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
                        torch.arange(el)[:, None] + i
                        for i in range(
                            self.future_traj_window[0], self.future_traj_window[1]
                        )
                    ]
                ),
                dim=1,
            )
            q = torch.clip(q, 0, el - 1)
            querys.append(q)
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
        num_zero_histories = []
        eds = []
        img_fnss = []
        qs = []
        future_img_fnss = []

        for i, (query, key_img, ed, fts) in tqdm(
            enumerate(
                zip(
                    self.querys,
                    self.keys_image,
                    self._episode_dirs,
                    self.future_traj_step,
                )
            ),
            desc="organize_file_names",
        ):
            for q, f in zip(query, fts):
                img_fns = []
                for img_idx in q:
                    img_fns.append(key_img[img_idx] if img_idx >= 0 else None)
                future_img_fns = []
                img_idx = random.choice(f)
                future_img_fns.append(key_img[img_idx] if img_idx >= 0 else None)
                num_zero_history = (q < 0).sum()
                # num_zero_history_list.append(int(num_zero_history))
                num_zero_histories.append(int(num_zero_history))
                eds.append(ed)
                img_fnss.append(img_fns[0])
                qs.append(int(q))
                future_img_fnss.append(future_img_fns[0])
                # values.append(
                #     [
                #         int(num_zero_history),
                #         ed,
                #         img_fns[0],
                #         int(q),
                #         # episode_length=self._episode_length[i],
                #         # future_traj_candidate=f,
                #         future_img_fns,
                #     ],
                # )
        num_zero_histories = torch.IntTensor(num_zero_histories)
        qs = torch.IntTensor(qs)

        img_fnss = np.array(img_fnss)
        future_img_fnss = np.array(future_img_fnss)
        eds = np.array(eds)

        return num_zero_histories, qs, img_fnss, eds, future_img_fnss

    def __len__(self):
        return len(self.img_fnss)

    def calc_fk(self, joint):
        """
        get end effector's position and orientation in world coordinate system
        Parameter:
        - obs(dict): observations with joints status
        Returns:
        - obs(dict): position and orientation will be stored in obs
        """
        ee_position, ee_orientation = [], []
        position, orientation = [], []
        for i in range(len(joint)):
            p.resetJointStatesMultiDof(
                self.panda, range(9), [[pos] for pos in joint[i]]
            )
            pos, orn = p.getLinkState(self.panda, self.panda_ee_index)[:2]
            pos = list(pos)
            pos.append(0)
            position.append(torch.FloatTensor(pos))
            orientation.append(torch.FloatTensor(orn))
        ee_position.append(torch.stack(position))
        ee_orientation.append(torch.stack(orientation))
        return torch.stack(ee_position), torch.stack(ee_orientation)

    def __getitem__(self, idx):
        num_zero_h = self.num_zero_histories[idx]
        q = self.qs[idx]
        img_fn = self.img_fnss[idx]
        ed = self.eds[idx]
        f_img_fn = self.future_img_fnss[idx]
        # value = self.values[idx]
        view = "front_1"
        img, wrist_img = self.get_image(img_fn, view)
        future_img, future_id = self.get_future_image(f_img_fn, view)
        # lang = self.get_language_instruction()
        ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, future_ee_traj = self.get_ee_data(
            ed,
            q.unsqueeze(0),
            num_zero_h,
        )
        pos, orn = self.calc_fk(joint)
        _data = {}
        _data["obs"] = {
            "agentview_image": img.permute(1, 2, 0).unsqueeze(0).float(),
            "robot0_eye_in_hand_image": wrist_img.permute(1, 2, 0).unsqueeze(0).float(),
            "robot0_eef_pos": torch.from_numpy(future_ee_traj[view][:1])
            .float()
            .flatten()
            .unsqueeze(0),
            "robot0_eef_quat": orn[0],
            "robot0_eef_pos_future_traj": torch.from_numpy(future_ee_traj[view][1:])
            .float()
            .flatten()
            .unsqueeze(0),
        }
        _data["goal_obs"] = {
            "agentview_image": future_img.permute(1, 2, 0).unsqueeze(0).float(),
            "robot0_eef_pos": torch.from_numpy(future_ee_traj[view][:1])
            .float()
            .flatten()
            .unsqueeze(0),
            "robot0_eef_pos_future_traj": torch.from_numpy(future_ee_traj[view][1:])
            .float()
            .flatten()
            .unsqueeze(0),
        }
        _data["actions"] = torch.cat(
            (
                torch.from_numpy(ee_pos_cmd),
                torch.from_numpy(ee_rot_cmd),
                torch.from_numpy(gripper_cmd),
            ),
            dim=-1,
        ).float()
        _data["rewards"] = torch.zeros(1).float()
        _data["dones"] = torch.zeros(1).float()
        return _data

    def get_future_image(self, future_img_fn, c_v):
        # future_id = random.choice(range(len(value["future_img_fns"])))
        # future_img_fn = value[4][0]
        future_img = self.transforms(
            np.array(Image.open(future_img_fn.replace(self._cam_view[0], c_v)))
        )[:3, :, :]
        return future_img, 0

    def get_image(self, img_fn, c_v):
        """
        This function generates the step for current frame and history frames
        Parameters:
        - episode_length(list of int): number of episode lengths for each episode
        Returns:
        - keys(list of tensors): history steps for each data
        """
        # imgs = []
        # for img_fn in img_fns:
        #     img_multi_view = {}
        #     for c_v in self._cam_view:
        #         img_multi_view[c_v] = self.transforms(
        #             np.array(Image.open(img_fn.replace(self._cam_view[0], c_v)))
        #             if img_fn != None
        #             else np.zeros_like(Image.open(img_fns[-1]))
        #         )[:3, :, :]
        #     imgs.append(img_multi_view)
        # img_fn = img_fns[0]
        img = self.transforms(
            np.array(Image.open(img_fn.replace(self._cam_view[0], "front_1")))
        )[:3, :, :]
        wrist_img = self.transforms(
            np.array(Image.open(img_fn.replace(self._cam_view[0], "wrist")))
        )[:3, :, :]
        # plt.subplot(1, 2, 1)
        # plt.imshow(img.permute(1, 2, 0))
        # plt.subplot(1, 2, 2)
        # plt.imshow(wrist_img.permute(1, 2, 0))
        # plt.show()
        return img, wrist_img

    def randomly_select_camview(self, imgs):
        assert imgs.size(0) == 1
        imgs = imgs[0]
        idx = random.randint(0, len(self._cam_view) - 2)
        img = imgs[idx]
        wrist_img = imgs[-1]
        return img, wrist_img, idx

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
        len_raw_data = len(raw_data)
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
        future_ee_traj = {}
        for cv in self._cam_view:
            if cv != "wrist":
                future_ee_traj[cv] = np.zeros([self.future_traj_horizon, 3])
                future_ee_traj[cv][
                    0 : clamp(
                        end_idx + self.future_traj_horizon,
                        start_idx,
                        len_raw_data,
                    )
                    - start_idx,
                    :,
                ] = raw_raw_data.loc[
                    start_idx : clamp(
                        end_idx + self.future_traj_horizon - 1,
                        start_idx,
                        len_raw_data,
                    ),
                    [f"{cv}_ee_pos_{ax}" for ax in ["x", "y", "z"]],
                ].to_numpy()

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
        )
        gripper_data = (
            raw_data.loc[start_idx:end_idx, "gripper_closedness_commanded"]
            .to_numpy()
            .reshape(-1, 1)
        )
        gripper_cmd = np.vstack((gripper_cmd, gripper_data))
        return ee_pos_cmd, ee_rot_cmd, gripper_cmd, joint, future_ee_traj

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


def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


import h5py


# @profile
def main():
    with h5py.File(
        "/nfs1/factorworld_dataset/version_pick_only_fa_/Panda/data.h5", "r"
    ) as file:
        a = file["data"]
        print(a)
    data = pd.read_hdf(
        "/nfs1/factorworld_dataset/version_pick_only_fa_/Panda/data.h5",
        key="data",
    )
    args = {
        "data_path": "/home/io011/nfs/factorworld_dataset/version_0.0_rdt_rc_rwt_reot_rl_rd_rdp_/Panda",
        "time_sequence_length": 1,
        "H": [20, 60],
        "predicting_next_ts": True,
        "num_train_episode": 10,
        "num_val_episode": 20,
        "cam_view": ["front_0", "front_1", "front_2", "wrist"],
    }
    train_dataset, val_dataset = build_dataset(
        data_path=args["data_path"],
        time_sequence_length=args["time_sequence_length"],
        predicting_next_ts=args["predicting_next_ts"],
        cam_view=args["cam_view"],
        H=args["H"],
        language_embedding_size=512,
        future_traj_horizon=21,
    )
    dataloader = DataLoader(train_dataset, 24, num_workers=0, shuffle=False)
    posx = []
    posy = []
    posz = []
    rotx = []
    roty = []
    rotz = []
    for b in tqdm(dataloader):
        pass
        print(13123)


if __name__ == "__main__":
    main()

    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     plt.imshow(b["goal_obs"]["agentview_image"][i, 0] * 255)
    #     fet = b["obs"]["robot0_eef_pos_future_traj"][i, 0].view(
    #         -1, 3
    #     ) * torch.tensor([128, -100, 0]) + torch.tensor([160, 128, 0])
    #     plt.scatter(fet[:1, 0], fet[:1, 1], c="r")
    #     plt.plot(fet[:, 0], fet[:, 1], c="b")
    # plt.show()
