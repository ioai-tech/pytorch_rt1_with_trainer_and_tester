import argparse
import pybullet as p
import pybullet_data as pdata
import numpy as np
import argparse
import random
import time
import os
import cv2
from sim_env import get_cam_view_img
from skvideo.io import vwrite
import torch
from gym import spaces
from collections import OrderedDict
import copy
from func_timeout import func_set_timeout
import func_timeout
from tqdm import trange
import json

from maruya24_rt1.transformer_network import TransformerNetwork
from maruya24_rt1.transformer_network_test_set_up import state_space_list
from maruya24_rt1.tokenizers.utils import batched_space_sampler
from maruya24_rt1.tokenizers.utils import np_to_tensor
from sim_env import *
import util.misc as utils
import torch.nn.functional as F
from torchvision import transforms


class SimTester:
    def __init__(self, task_name, max_step=200, device="cuda", sim_interval=0.005):
        p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraPitch=-20,
            cameraYaw=180,
            cameraTargetPosition=[0, 0, 0.6],
        )
        p.setAdditionalSearchPath(pdata.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.task_env = globals()[TASKS[task_name]]()
        self.max_step = max_step
        self.device = torch.device(device)
        self.sim_interval = sim_interval

    def test_step(self, delta_ee_pos, delta_ee_rot, gripper_cmd, cam_views):
        try:
            self.execute_action(delta_ee_pos, delta_ee_rot, gripper_cmd, cam_views)
        except func_timeout.exceptions.FunctionTimedOut:
            # print("robot stuck")
            pass
        self.update_obs(cam_views)

    @func_set_timeout(
        0.5
    )  # Timeout set to handle cases where the robot may get stuck, allowing the next action to execute
    def execute_action(
        self, delta_ee_pos, delta_ee_rot, gripper_cmd, cam_views, relative=True
    ):
        # Calculate the new end-effector pose based on relative or absolute movements
        if relative:
            # Obtain the current end-effector pose
            last_ee_pose = p.getLinkState(
                self.task_env.robot.robot_id, self.task_env.robot.ee_index
            )[0:2]

            # Calculate the target end-effector pose based on the relative movement
            target_ee_pose = p.multiplyTransforms(
                last_ee_pose[0],
                last_ee_pose[1],
                delta_ee_pos,
                p.getQuaternionFromEuler(delta_ee_rot),
            )
        else:
            # Handling for absolute movements, if implemented
            raise NotImplementedError("Absolute movement handling not implemented yet")

        # Calculate inverse kinematics to obtain joint positions for the target end-effector pose
        tar_j_pos = self.task_env.robot.calc_ik([target_ee_pose[0], target_ee_pose[1]])

        # Move the robot's arm to the calculated joint positions
        self.task_env.robot.move_arm(tar_j_pos)

        # Control the gripper based on the provided command
        self.task_env.robot.move_gripper(gripper_cmd[0])

        # Ensure the robot reaches the target joint positions within a threshold
        while not (self.task_env.robot.is_j_arrived(tar_j_pos, threshold=1e-3)):
            p.stepSimulation()  # Step the simulation
            time.sleep(self.sim_interval)  # Wait for the simulation to progress

    def reset_tester(self, cam_views):
        self.task_env.reset_env()
        self.reset_obs(cam_views)
        self.gripper_triggered = False
        self.episode_succ = [False, False]

    def reset_obs(self, cam_views):
        self.imgs = [
            torch.zeros(1, 3, 256 * len(cam_views), 320)
        ] * self.time_sequence_length
        self.joints = [torch.zeros(1, 9)] * self.time_sequence_length

    def update_obs(self, cam_views):
        # Obtain observation images from specified camera views
        img = self.get_obs_img(cam_views)

        # Remove the oldest image from the image collection and add the new observation image
        self.imgs.pop(0)
        self.imgs.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0))

        # Retrieve the current joint status
        joint = self.get_joint_status()

        # Remove the oldest joint status and add the current joint status to the collection
        self.joints.pop(0)
        self.joints.append(joint.unsqueeze(0))

    def get_joint_status(self):
        """
        Obtain the current joint status of the robot.

        Returns:
        - torch.tensor: Tensor containing the current joint states of the robot.
        """
        return torch.tensor(
            [
                s[0]
                for s in p.getJointStates(
                    self.task_env.robot.robot_id,
                    list(range(self.task_env.robot.arm_dof)) + [9, 10],
                )
            ]
        )

    def get_obs_img(self, cam_views):
        """
        Generate observation images from specified camera views.

        Args:
        - cam_views (list): List of camera views.

        Returns:
        - np.ndarray: Observation image array.
        """
        imgs = []
        for cview in cam_views:
            imgs.append(
                get_cam_view_img(
                    cview, self.task_env.robot.robot_id, self.task_env.robot.ee_index
                )
            )
        cur_img = np.concatenate(imgs, axis=0)
        return cur_img / 255.0

    def check_episode_succ(self):
        """
        Check if the episode is successful based on end-effector and target object positions.

        Returns:
        - float: Difference in z-axis positions between the end-effector and the target object.
        """
        ee_z_pos = p.getLinkState(
            self.task_env.robot.robot_id, self.task_env.robot.ee_index
        )[0][2]
        tar_obj_z_pos = self.task_env.tar_obj_pose[0][2]
        contact_points = p.getContactPoints(
            self.task_env.robot.robot_id, self.task_env.tar_obj
        )
        if abs(ee_z_pos - tar_obj_z_pos) < 0.035 and len(contact_points) > 0:
            self.episode_succ[0] = True
        if ee_z_pos - tar_obj_z_pos > 0.08 and len(contact_points) > 0:
            self.episode_succ[1] = True
        return ee_z_pos - tar_obj_z_pos


def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


class Evaluator:
    def __init__(self, task_name) -> None:
        self.sim_tester = SimTester(task_name)
        self.pandaEndEffectorIndex = 11
        args = load_config_from_json("train_config.json")
        self.args = args

    def write_results(self, fn, episode_succ):
        with open(fn, "a") as f:
            f.write(str(int(episode_succ[0])) + str(int(episode_succ[1])) + "\n")
        f.close()

    def calc_fk(self, obs):
        """
        get end effector's position and orientation in world coordinate system
        Parameter:
        - obs(dict): observations with joints status
        Returns:
        - obs(dict): position and orientation will be stored in obs
        """
        ee_position, ee_orientation = [], []
        for joint in obs["joint_position"]:
            position, orientation = [], []
            for i in range(len(joint)):
                p.resetJointStatesMultiDof(
                    self.sim_tester.task_env.robot.robot_id,
                    range(9),
                    [[pos] for pos in joint[i]],
                )
                pos, orn = p.getLinkState(self.sim_tester.task_env.robot.robot_id, 11)[
                    :2
                ]
                pos = list(pos)
                pos.append(0)
                position.append(torch.FloatTensor(pos))
                orientation.append(torch.FloatTensor(orn))
            ee_position.append(torch.stack(position))
            ee_orientation.append(torch.stack(orientation))
        obs["position"] = torch.stack(ee_position).to(self.device)
        obs["orientation"] = torch.stack(ee_orientation).to(self.device)
        del obs["joint_position"]
        return obs

    def inference(self, network, imgs, joints):
        """
        Perform inference using the provided neural network on input observations.

        Args:
        - network (NeuralNetwork): Neural network model for inference.
        - imgs (list): List of observation images.
        - joints (list): List of joint positions.

        Returns:
        - list: List containing the inferred actions related to termination, world vector, rotation delta, and gripper closedness.
        """
        # Batched space sampling for the network state
        network_state = batched_space_sampler(network._state_space, batch_size=1)
        network_state = np_to_tensor(network_state)  # Convert np.ndarray to tensor
        for k, v in network_state.items():
            network_state[k] = torch.zeros_like(v).to(self.device)

        output_actions = []
        obs = dict()
        obs["image"] = torch.stack(imgs, dim=1).to(
            self.device
        )  # Stack images on the device
        obs["natural_language_embedding"] = (
            torch.ones(1, network._time_sequence_length) * 3
        )
        obs["natural_language_embedding"] = (
            obs["natural_language_embedding"].long().to(self.device)
        )  # Prepare language embedding tensor

        # Stack joint positions and adjust dimensions
        obs["joint_position"] = torch.stack(joints).permute(1, 0, 2)

        # If the network uses proprioception, calculate forward kinematics
        if network.using_proprioception:
            obs = self.calc_fk(obs)

        with torch.no_grad():
            for i_ts in range(network._time_sequence_length):
                ob = utils.retrieve_single_timestep(obs, i_ts)
                output_action, network_state = network(ob, network_state)
                output_actions.append(output_action)

            # Retrieve the final inferred action
            action = output_actions[-1]

        # Move the action to the CPU and convert it to a list for return
        action = utils.dict_to_device(action, torch.device("cpu"))
        return [
            action["terminate_episode"].flatten().tolist(),
            action["world_vector"].flatten().tolist(),
            action["rotation_delta"].flatten().tolist(),
            action["gripper_closedness_action"].flatten().tolist(),
        ]

    def val(
        self,
        resume_from_checkpoint,
        ckpt_dir,
        proc_name,
        cam_views,
        test_num,
        results_fn,
        epoch,
        gpu_name,
        using_pos_and_orn,
        tar_obj_pos_rot,
    ):
        self.resume_from_checkpoint = resume_from_checkpoint

        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
        self._action_space = checkpoint["action_space"]
        network_configs = self.args["network_configs"]
        network_configs["time_sequence_length"] = self.args["time_sequence_length"]
        network_configs["num_encoders"] = len(self.args["cam_view"])
        network_configs["using_proprioception"] = self.args["using_proprioception"]
        network_configs["token_embedding_size"] = network_configs[
            "token_embedding_size_per_image"
        ] * len(self.args["cam_view"])
        del network_configs["token_embedding_size_per_image"]
        network_configs["input_tensor_space"] = state_space_list()[0]
        network_configs["output_tensor_space"] = self._action_space
        self.network = TransformerNetwork(**network_configs)
        setattr(
            self.sim_tester, "time_sequence_length", self.network._time_sequence_length
        )
        try:
            local_rank = os.environ["LOCAL_RANK"]
            torch.cuda.set_device(int(local_rank))
        except:
            pass
        self.device = torch.device("cuda")
        self.network.to(self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.network.eval()
        try:
            os.mkdir(os.path.join(ckpt_dir, "vids"))
        except:
            pass
        lift_count = 0
        for idx in range(test_num):
            vid_fn = os.path.join(
                ckpt_dir,
                "vids",
                "e"
                + str(epoch)
                + "_i"
                + str(idx)
                + "_p"
                + str(proc_name)
                + "_g"
                + str(gpu_name)
                + ".mp4",
            )
            self.sim_tester.reset_tester(cam_views)
            # self.sim_tester.task_env.reset_tar_obj(
            #     tar_pos_rot=tar_obj_pos_rot, random_pos_rot=False
            # )
            time.sleep(1)
            self.sim_tester.update_obs(cam_views)
            vid = []
            for _ in trange(self.sim_tester.max_step):
                imgs = self.sim_tester.imgs
                joints = self.sim_tester.joints
                (
                    terminate_episode,
                    delta_ee_pos,
                    delta_ee_rot,
                    gripper_cmd,
                ) = self.inference(self.network, imgs, joints)
                self.sim_tester.test_step(
                    delta_ee_pos, delta_ee_rot, gripper_cmd, cam_views
                )
                distance_to_target = self.sim_tester.check_episode_succ()
                frame = copy.deepcopy(imgs[-1][0].permute(1, 2, 0).numpy() * 255)
                frame_rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
                text = (
                    "epoch: "
                    + str(epoch)
                    + " dis: "
                    + str(round(distance_to_target, 3))
                )
                cv2.putText(
                    frame_rgb, text, (0, 20), cv2.FONT_ITALIC, 0.75, (0, 0, 0), 2
                )
                if self.sim_tester.episode_succ[0] == True:
                    cv2.putText(
                        frame_rgb,
                        "grabbed",
                        (0, 40),
                        cv2.FONT_ITALIC,
                        0.75,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame_rgb,
                        "not grabbed",
                        (0, 40),
                        cv2.FONT_ITALIC,
                        0.75,
                        (0, 0, 255),
                        2,
                    )
                if self.sim_tester.episode_succ[1] == True:
                    cv2.putText(
                        frame_rgb,
                        "lifted",
                        (0, 60),
                        cv2.FONT_ITALIC,
                        0.75,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        frame_rgb,
                        "not lifted",
                        (0, 60),
                        cv2.FONT_ITALIC,
                        0.75,
                        (0, 0, 255),
                        2,
                    )
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                vid.append(frame_rgb)
                if self.sim_tester.episode_succ[1] == True:
                    lift_count += 1
                    if lift_count >= 5:
                        print("lifted")
                        break
            self.sim_tester.check_episode_succ()
            self.write_results(results_fn, self.sim_tester.episode_succ)
            vwrite(vid_fn, vid)


def test_loss(
    ckpt_dir,
    evaluator,
    cam_views,
    model_name,
    proc_name,
    epoch,
    gpu_name,
    using_pos_and_orn,
    num_threads,
):
    cam_views = cam_views.split("_")
    fn = os.path.join(ckpt_dir, str(epoch) + ".txt")
    fn_episodes = os.path.join(ckpt_dir, str(epoch) + "_val_episodes.txt")
    if num_threads > 0:
        with open(fn_episodes, "r") as f:
            val_episode_dirs = f.readlines()

        val_episode_dir = val_episode_dirs[
            int(gpu_name) * int(num_threads) + int(proc_name)
        ].strip()

        visual_data_filename_raw = f"{val_episode_dir}result_raw.csv"
        raw_raw_data = pd.read_csv(visual_data_filename_raw)
        # print(raw_raw_data.iloc[0])
        tar_obj_pos_rot = [
            raw_raw_data.iloc[0]["tar_obj_pose_x"],
            raw_raw_data.iloc[0]["tar_obj_pose_y"],
            raw_raw_data.iloc[0]["tar_obj_pose_rx"],
            raw_raw_data.iloc[0]["tar_obj_pose_ry"],
            raw_raw_data.iloc[0]["tar_obj_pose_rz"],
            raw_raw_data.iloc[0]["tar_obj_pose_rw"],
        ]
    else:
        tar_obj_pos_rot = [0, 0, 0, 0, 0, 0, 0]
    utils.set_seed()
    evaluator.val(
        os.path.join(ckpt_dir, model_name),
        ckpt_dir=ckpt_dir,
        proc_name=proc_name,
        cam_views=cam_views,
        test_num=1,
        results_fn=fn,
        epoch=epoch,
        gpu_name=gpu_name,
        using_pos_and_orn=using_pos_and_orn,
        tar_obj_pos_rot=tar_obj_pos_rot,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="pick")
    parser.add_argument("-d", "--ckpt_dir", type=str, default="logs")
    parser.add_argument("-c", "--cam_views", type=str, default="topdown_wrist")
    parser.add_argument("-m", "--model_name", type=str, default="99-checkpoint.pth")
    parser.add_argument("-p", "--proc_name", type=int, default=200)
    parser.add_argument("-e", "--epoch_num", type=str, default="0")
    parser.add_argument("-g", "--gpu_name", type=str, default="3")
    parser.add_argument("-u", "--using_pos_and_orn", type=int, default=False)
    parser.add_argument("-n", "--num_threads", type=int, default=0)

    args = parser.parse_args()
    print(args.proc_name)
    # exit()
    task_name = args.task
    evaluator = Evaluator(task_name)
    test_loss(
        ckpt_dir=args.ckpt_dir,
        evaluator=evaluator,
        cam_views=args.cam_views,
        model_name=args.model_name,
        proc_name=args.proc_name,
        epoch=args.epoch_num,
        gpu_name=args.gpu_name,
        using_pos_and_orn=bool(args.using_pos_and_orn),
        num_threads=args.num_threads,
    )
# python IO_evaluator.py -d /mnt/logs_1/1698325798/999-checkpoint.pth -c front wrist -m 999-checkpoint.pth
