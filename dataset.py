import os
import re
import json
import glob
import random
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
from tqdm import tqdm as tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import cProfile
import pstats


def build_dataset(
    data_path,
    time_sequence_length=18,
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
    # robot = data_path.split("/")[-1].split("_")[0]
    # with open(os.path.join(data_path, cam_view[0], "dataset_info.json"), "r") as f:
    #     info = json.load(f)
    # episode_length = info["episode_length"]
    # episode_dirs = sorted(glob.glob(data_path + "/" + cam_view[0] + "/*/"))
    # assert len(episode_dirs) == len(
    #     episode_length
    # ), "length of episode directories and episode length not equal, check dataset's dataset_info.json"
    # perm_indice = torch.randperm(len(episode_dirs)).tolist()
    # dirs_lengths = dict(
    #     episode_dirs=np.array(episode_dirs)[perm_indice],
    #     episode_length=np.array(episode_length)[perm_indice],
    # )
    episode_dirs = glob.glob(data_path + "/*")
    # train_episode_dirs = dirs_lengths["episode_dirs"][:num_train_episode]
    # train_episode_length = dirs_lengths["episode_length"][:num_train_episode]
    # val_episode_dirs = dirs_lengths["episode_dirs"][
    #     num_train_episode : num_train_episode + num_val_episode
    # ]
    # val_episode_length = dirs_lengths["episode_length"][
    #     num_train_episode : num_train_episode + num_val_episode
    # ]
    train_dataset = PreTrainingDataset(
        data_dirs=episode_dirs[:num_train_episode],
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_views=cam_view,
    )
    val_dataset = PreTrainingDataset(
        data_dirs=episode_dirs[num_train_episode : num_train_episode + num_val_episode],
        time_sequence_length=time_sequence_length,
        predicting_next_ts=predicting_next_ts,
        cam_views=cam_view,
    )
    return train_dataset, val_dataset


class Frame:
    def __init__(self) -> None:
        pass


class PreTrainingDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        time_sequence_length=16,
        positive_margin=10,
        negative_margin=30,
        predicting_next_ts=True,
        w=256,
        h=320,
        cam_views=["rgb", "cam_fisheye"],
    ):
        self.data_dirs = data_dirs
        self._time_sequence_length = time_sequence_length
        self.cam_views = cam_views
        self._positive_margin = positive_margin
        self._negative_margin = negative_margin

        self.pattern = re.compile(r"(\d+)_(\d+).png")
        self.episodes_stats, self.datas = self.calc_episode_stats()
        self.w = w
        self.h = h
        self.predicting_next_ts = predicting_next_ts
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.w, self.h), antialias=True),
                # transforms.RandomRotation(degrees=(-5, 5)),
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            ]
        )
        with open("class.json", "r") as json_file:
            self.classes = json.load(json_file)

    def calc_episode_stats(self):
        episodes_stats = {}
        datas = []
        for ddr in tqdm(self.data_dirs):
            if ddr.endswith("json") or ddr.endswith("yml"):
                continue
            # ee_data = pd.read_csv(os.path.join(ddr, "result.csv"))
            # timestamp = list(ee_data["timestamp"])
            rgb_imgs_fns = os.listdir(os.path.join(ddr, "rgb"))
            rgb_imgs_fns.sort(key=lambda x: x[-23:-4])
            timestamp = []
            for fn in rgb_imgs_fns:
                timestamp.append(int(fn[-23:-4]))
            imgfn_idx_ts_pairs = []
            for ts in timestamp:
                paired_img_fn = None
                for img_fn in rgb_imgs_fns:
                    if str(ts) in img_fn:
                        paired_img_fn = img_fn
                        pair = dict(
                            ddr=ddr,
                            ts=ts,
                            img_fn=paired_img_fn,
                            idx=int(paired_img_fn.split("raw")[-1].split("_")[1])
                            # idx=int(
                            #     re.search(self.pattern, paired_img_fn).group(1)
                            # ),
                        )
                        imgfn_idx_ts_pairs.append(pair)
                        datas.append(pair)
                        rgb_imgs_fns.remove(img_fn)
                        break
            episodes_stats[ddr] = imgfn_idx_ts_pairs
        return episodes_stats, datas

    def generate_fn_lists(self):
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
        datas = []
        for episode_dir, episode in self.episodes_stats.items():
            for cam_view, values in episode.items():
                if cam_view != "timestamp":
                    for idx, fn in values.items():
                        datas.append(
                            dict(
                                episode_dir=episode_dir,
                                cam_view=cam_view,
                                idx=idx,
                                fn=fn,
                            )
                        )

        return datas

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
        return len(self.datas)

    def __getitem__(self, idx):
        # profiler = cProfile.Profile()
        # profiler.enable()
        value = self.datas[idx]
        # profiler = cProfile.Profile()
        # profiler.enable()
        sample_obs, sample_action = self.get_item(value)
        # profiler.disable()
        # # 将结果保存到一个文件
        # profiler.dump_stats("profile_results.stats")

        # # 使用 pstats 加载和分析结果
        # stats = pstats.Stats("profile_results.stats")
        # # 排序：可选的参数包括 'cumulative', 'time', 'calls' 等
        # stats.sort_stats("cumulative")
        # with open("profile_results.txt", "w") as f:
        #     stats = pstats.Stats(profiler, stream=f)
        #     stats.strip_dirs()
        #     stats.sort_stats("cumulative")
        #     stats.print_stats()

        return sample_obs, sample_action

    def get_item(self, value):
        values = self.episodes_stats[value["ddr"]]
        a_idxs = [v["idx"] for v in values]
        # episode_dict = {idx: v for idx, v in zip(a_idxs, values)}
        min_idx = min(a_idxs)
        max_idx = max(a_idxs)
        idxs = torch.arange(
            value["idx"] - self._time_sequence_length + 1, value["idx"] + 1
        )
        idxs = torch.where(idxs < min_idx, torch.tensor(-1), idxs)
        imgs = self.get_image(value, idxs)
        lang = self.get_language_instruction(value)
        ee_pos_cmd, ee_rot_cmd, gripper_cmd = self.get_ee_data(value, idxs, min_idx)
        terminate_episode = self.get_episode_status(
            min_idx=min_idx,
            max_idx=max_idx,
            query_index=idxs,
            pad_step_num=torch.sum(torch.eq(idxs, -1)),
        )
        sample_obs = {
            "image": imgs.float(),
            "natural_language_embedding": lang.long(),
        }
        sample_action = {
            "world_vector": ee_pos_cmd.float(),
            "rotation_delta": ee_rot_cmd.float(),
            "gripper_closedness_action": gripper_cmd.float(),
            "terminate_episode": torch.from_numpy(terminate_episode).argmax(-1),
        }
        return sample_obs, sample_action

    def get_image(self, value, idxs):
        # TODO: read image not correspond to new dataset

        cv_fn_pair = {}
        for c_v in self.cam_views:
            cv_fn_pair[c_v] = os.listdir(os.path.join(value["ddr"], c_v))
        imgs = []
        for idx in idxs:
            if idx == -1:
                imgs_multi_cam = []
                for c_v in self.cam_views:
                    imgs_multi_cam.append(self.transform(np.zeros((self.w, self.h, 3))))
            else:
                # img_fns = os.listdir(os.path.join(
                #                             episode_dict[int(idx)]["ddr"],
                #                             c_v,)
                imgs_multi_cam = []
                for c_v in self.cam_views:
                    fn = list(
                        filter(lambda img: str(int(idx)) in img, cv_fn_pair[c_v])
                    )[0]
                    # fn = cv_fn_pair[c_v]
                    # fn_ = fn.split("_")
                    # fn = ""
                    # for segs in fn_[:-2]:
                    #     fn += segs + "_"
                    # fn = fn + str(int(idx)) + "_" + fn_[-1]
                    imgs_multi_cam.append(
                        self.transform(
                            np.array(
                                Image.open(
                                    os.path.join(
                                        value["ddr"],
                                        c_v,
                                        fn,
                                    )
                                )
                            )[:, :, :3]
                            # np.zeros((self.w, self.h, 3))
                        )
                    )
            imgs.append(torch.cat(imgs_multi_cam, dim=1))
        # for i in range(len(idxs)):
        #     plt.subplot(1, len(idxs), i + 1)
        #     plt.imshow(imgs[i].permute(1, 2, 0))
        # plt.show()
        return torch.stack(imgs)

    def get_ee_data(self, value, idxs, min_idx):
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
        ee_data = pd.read_csv(os.path.join(value["ddr"], "result.csv"))
        if self.predicting_next_ts:
            """
            if predicting next timestep's results, then we shift first column to last column
            """
            first_row = ee_data.iloc[0]
            ee_data = ee_data.iloc[1:]
            ee_data = pd.concat([ee_data, first_row.to_frame().T], ignore_index=True)

        ee_pos_cmd = []
        ee_rot_cmd = []
        gripper_cmd = []
        for idx in idxs:
            if idx == -1:
                ee_pos_cmd.append(torch.zeros(3))
                ee_rot_cmd.append(torch.zeros(3))
                gripper_cmd.append(torch.zeros(1))
            else:
                ee_pos_cmd.append(
                    torch.from_numpy(
                        ee_data.loc[
                            int(idx - min_idx),
                            [f"ee_command_position_{ax}" for ax in ["x", "y", "z"]],
                        ].to_numpy()
                    )
                )
                ee_rot_cmd.append(
                    torch.from_numpy(
                        ee_data.loc[
                            int(idx - min_idx),
                            [f"ee_command_rotation_{ax}" for ax in ["x", "y", "z"]],
                        ].to_numpy()
                    )
                )
                gripper_cmd.append(
                    torch.from_numpy(
                        ee_data.loc[
                            int(idx - min_idx),
                            [f"gripper_closedness_commanded"],
                        ].to_numpy()
                    )
                )

        return (
            torch.stack(ee_pos_cmd),
            torch.stack(ee_rot_cmd),
            torch.stack(gripper_cmd),
        )

    def get_language_instruction(self, value):
        """
        since we are only training single-task model, this language embedding is set as constant.
        modify it to language instructions if multi-task model is training.
        it seems that google directly loads embedded language instruction from its language model
        this results in our loading a language embedding instead of language sentence
        """
        lan = value["ddr"].split("/")[-1]
        lan = self.classes[re.match(r"^\d{8}_(\w+)_\d+_\d+$", lan).group(1)]
        return torch.ones(self._time_sequence_length) * lan

    def get_episode_status(self, min_idx, max_idx, query_index, pad_step_num):
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
                [
                    i == min_idx,
                    i not in [min_idx, max_idx],
                    i == max_idx,
                    0,
                ],
                dtype=np.int32,
            )
            episode_status = np.vstack((episode_status, status))
        if pad_step_num > 0:
            episode_status[pad_step_num] = np.array([1, 0, 0, 0])
        # print(torch.tensor(episode_status).argmax(-1)[-1])
        # if torch.tensor(episode_status).argmax(-1)[-1] == 2:
        #     pass
        return episode_status


def extract_keywords_from_folder_names(folder_path):
    """
    遍历指定文件夹下的所有子文件夹，并提取子文件夹名中的关键词。

    :param folder_path: 需要遍历的文件夹路径。
    :return: 包含所有关键词的列表。
    """
    keywords = []
    pattern = r"^\d{8}_(\w+)_\d+_\d+$"  # 匹配类似 "20231101_apple_0_12" 的模式

    # 遍历文件夹
    for item in os.listdir(folder_path):
        # 检查是否为文件夹
        if os.path.isdir(os.path.join(folder_path, item)):
            # 使用正则表达式提取关键词
            match = re.match(pattern, item)
            if match:
                keyword = match.group(1)
                keywords.append(keyword)
    c = Counter(keywords)
    ks = list(c.keys())
    keys = {}
    for i, k in enumerate(ks):
        keys[k] = i
    with open("class.json", "w") as json_file:
        json.dump(keys, json_file, indent=4)
    return keys


if __name__ == "__main__":

    def load_config_from_json(json_path):
        with open(json_path, "r") as f:
            config = json.load(f)
        return config

    args = load_config_from_json("train_config.json")
    train_dataset, val_dataset = build_dataset(
        data_path=args["data_path"],
        time_sequence_length=args["time_sequence_length"],
        predicting_next_ts=args["predicting_next_ts"],
        num_train_episode=args["num_train_episode"],
        num_val_episode=args["num_val_episode"],
        cam_view=args["cam_view"],
        language_embedding_size=args["network_configs"]["language_embedding_size"],
    )

    dataloader = DataLoader(train_dataset, 1, num_workers=0, shuffle=False)
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
        pass
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
