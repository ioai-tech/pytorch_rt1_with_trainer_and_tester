import copy
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import time
import subprocess
import random
import json
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from gym import spaces
import pybullet as p
import pybullet_data as pdata

from tqdm import tqdm

import util.misc as utils
from IO_dataset_torch import build_dataset
from maruya24_rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from maruya24_rt1.transformer_network import TransformerNetwork
from maruya24_rt1.transformer_network_test_set_up import state_space_list


def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


class Trainer:
    def __init__(self, args):
        utils.set_seed()
        self.args = args
        self.args = utils.init_distributed_mode(self.args)
        self.checkpoint_dir, self.tensorboard_dir = self.make_log_dir(
            self.args["log_dir"]
        )
        if self.args["mode"] == "eval":
            self.args["num_val_episode"] = (
                self.args["num_eval_threads"] * self.args["world_size"]
            )
        self.train_dataset, self.val_dataset = build_dataset(
            data_path=self.args["data_path"],
            time_sequence_length=self.args["time_sequence_length"],
            predicting_next_ts=self.args["predicting_next_ts"],
            num_train_episode=self.args["num_train_episode"],
            num_val_episode=self.args["num_val_episode"],
            cam_view=self.args["cam_view"],
            language_embedding_size=self.args["network_configs"][
                "language_embedding_size"
            ],
        )
        
        if self.args["distributed"]:
            self.sampler_train = DistributedSampler(self.train_dataset, shuffle=True)
            self.sampler_val = DistributedSampler(self.val_dataset, shuffle=False)

        self.args["checkpoint_dir"] = self.checkpoint_dir
        self.writer_train = SummaryWriter(self.tensorboard_dir, flush_secs=5)
        self.writer_val = SummaryWriter(self.tensorboard_dir + "_val", flush_secs=5)
        self._action_space = spaces.Dict(
            OrderedDict(
                [
                    ("terminate_episode", spaces.Discrete(4)),
                    (
                        "world_vector",
                        spaces.Box(low=-0.01, high=0.01, shape=(3,), dtype=np.float32),
                    ),
                    (
                        "rotation_delta",
                        spaces.Box(
                            low=-np.pi / 50,
                            high=np.pi / 50,
                            shape=(3,),
                            dtype=np.float32,
                        ),
                    ),
                    (
                        "gripper_closedness_action",
                        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    ),
                ]
            )
        )
        self.args["action_space"] = str(self._action_space)
        if utils.is_main_process():
            with open(
                os.path.join(self.checkpoint_dir, self.train_name + ".json"), "w"
            ) as json_file:
                json.dump(self.args, json_file)
            json_file.close()
        self.device = torch.device(self.args["device"])

        if self.args["using_proprioception"]:
            p.connect(p.DIRECT)
            p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
            p.setGravity(0, -9.8, 0)
            p.setAdditionalSearchPath(pdata.getDataPath())
            self.panda = p.loadURDF(
                "franka_panda/panda.urdf", [0, 0, 0.62], [0, 0, 0, 1], useFixedBase=True
            )
            self.panda_ee_index = 11

        self.train_step = 0
        self.val_step = 0

    def train(self):
        print("training")
        # Create dataloader based on distributed or single-machine settings
        if self.args["distributed"]:
            # Batch sampler for distributed training
            batch_sampler_train = torch.utils.data.BatchSampler(
                self.sampler_train, self.args["batch_size"], drop_last=True
            )
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler_train,
                num_workers=self.args["batch_size"],
            )
        else:
            # DataLoader for single-machine training
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args["batch_size"],
                num_workers=0,
                shuffle=True,
                drop_last=True,
            )

        # Initialize the TransformerNetwork based on specified configurations
        network_configs = self.args["network_configs"]
        # Modify network configuration based on specific settings
        network_configs["time_sequence_length"] = self.args["time_sequence_length"]
        network_configs["num_encoders"] = len(self.args["cam_view"])
        network_configs["token_embedding_size"] = network_configs[
            "token_embedding_size_per_image"
        ] * len(self.args["cam_view"])
        del network_configs["token_embedding_size_per_image"]
        network_configs["using_proprioception"] = self.args["using_proprioception"]
        network_configs["input_tensor_space"] = state_space_list()[0]
        network_configs["output_tensor_space"] = self._action_space
        network = TransformerNetwork(**network_configs)
        network.to(self.device)
        network_without_ddp = network

        # Load model weights, optimizer, scheduler settings, resume from checkpoints if specified
        if self.args["resume"]:
            checkpoint = torch.load(
                self.args["resume_from_checkpoint"], map_location="cpu"
            )
        total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print("number of model params:", total_params)
        total_size_bytes = total_params * 4
        # Parameter is in torch.float32，Each parameter takes 4 bytes
        total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
        print("model size: ", total_size_mb, " MB")

        # Configuration based on distributed or single-machine setup
        if self.args["distributed"]:
            # DistributedDataParallel setup
            network = torch.nn.parallel.DistributedDataParallel(
                network, device_ids=[self.args["gpu"]], find_unused_parameters=False
            )
            network_without_ddp = network.module
            optimizer = torch.optim.AdamW(
                network_without_ddp.parameters(), lr=self.args["lr"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, **self.args["scheduler_configs"]
            )
            if self.args["resume"]:
                network_without_ddp.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Single-machine setup
            optimizer = torch.optim.AdamW(network.parameters(), lr=self.args["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, **self.args["scheduler_configs"]
            )
            if self.args["resume"]:
                network.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # self.val(network_without_ddp, 0, self.val_dataset)
        # Training loop over epochs
        epoch_start = checkpoint["epoch"] if self.args["resume"] else 0
        for e in range(epoch_start, self.args["epochs"]):
            network.train()
            with tqdm(
                train_dataloader, dynamic_ncols=True, desc="train"
            ) as tqdmDataLoader:
                for _, (obs, action) in enumerate(tqdmDataLoader):
                    # Perform training steps
                    optimizer.zero_grad()
                    network_without_ddp.set_actions(
                        utils.dict_to_device(action, self.device)
                    )
                    network_state = batched_space_sampler(
                        network_without_ddp._state_space,
                        batch_size=self.args["batch_size"],
                    )
                    network_state = np_to_tensor(network_state)
                    if self.args["using_proprioception"]:
                        obs = self.calc_fk(obs)
                    output_actions, network_state = network(
                        utils.dict_to_device(obs, self.device),
                        utils.dict_to_device(network_state, self.device),
                    )

                    loss = network_without_ddp.get_actor_loss().mean()

                    loss.backward()
                    optimizer.step()

                    # Logging metrics during training
                    if utils.is_main_process():
                        # Log loss, epoch, and learning rate
                        self.writer_train.add_scalar(
                            tag="loss_ce",
                            global_step=self.train_step,
                            scalar_value=loss.cpu().data.numpy(),
                            walltime=time.time(),
                        )
                        self.writer_train.add_scalar(
                            tag="epoch",
                            global_step=self.train_step,
                            scalar_value=e,
                            walltime=time.time(),
                        )
                        self.writer_train.add_scalar(
                            tag="lr",
                            global_step=self.train_step,
                            scalar_value=optimizer.state_dict()["param_groups"][0][
                                "lr"
                            ],
                            walltime=time.time(),
                        )
                    self.train_step += 1
                    tqdmDataLoader.set_postfix(
                        ordered_dict={
                            "epoch": e,
                            "train_name": self.train_name[-5:],
                            "gpu_memory_used": str(
                                round(
                                    torch.cuda.max_memory_allocated() / (1024**3), 2
                                )
                            )
                            + " GB",
                            "loss": loss.item(),
                            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                        }
                    )

            # Perform validation at specified intervals
            if (e + 1) % self.args["val_interval"] == 0:
                checkpoint_filename = os.path.join(
                    self.checkpoint_dir, str(e) + "-checkpoint.pth"
                )
                checkpoint = {
                    "model_state_dict": network_without_ddp.state_dict()
                    if self.args["distributed"]
                    else network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "action_space": self._action_space,
                    "epoch": e,
                }
                utils.save_on_master(checkpoint, checkpoint_filename)
                if self.args["distributed"]:
                    # Barrier synchronization for distributed training
                    print(
                        f"Process {torch.distributed.get_rank()} has reached the end of epoch {e}."
                    )
                    torch.distributed.barrier()
                    self.val(
                        network_without_ddp=network_without_ddp,
                        epoch=e,
                        val_dataset=self.val_dataset,
                        sampler_val=self.sampler_val,
                    )
                    print(
                        f"Process {torch.distributed.get_rank()} has reached the end of val."
                    )
                    torch.distributed.barrier()
                else:
                    self.val(
                        network_without_ddp=network,
                        epoch=e,
                        val_dataset=self.val_dataset,
                    )
            scheduler.step()

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
                    self.panda, range(9), [[pos] for pos in joint[i]]
                )
                pos, orn = p.getLinkState(self.panda, self.panda_ee_index)[:2]
                pos = list(pos)
                pos.append(0)
                position.append(torch.FloatTensor(pos))
                orientation.append(torch.FloatTensor(orn))
            ee_position.append(torch.stack(position))
            ee_orientation.append(torch.stack(orientation))
        obs["position"] = torch.stack(ee_position).to(self.device)
        obs["orientation"] = torch.stack(ee_orientation).to(self.device)
        return obs

    @torch.no_grad()
    def val(self, network_without_ddp, epoch, val_dataset, sampler_val=None):
        # Create directories to store validation results if they don't exist
        if (
            not os.path.isdir(os.path.join(self.checkpoint_dir, "val_results"))
            and utils.is_main_process()
        ):
            os.mkdir(os.path.join(self.checkpoint_dir, "val_results"))

        # Set up dataloader based on distributed or single-machine settings
        if self.args["distributed"]:
            val_dataloader = DataLoader(
                val_dataset, batch_size=1, sampler=sampler_val, drop_last=False
            )
        else:
            val_dataloader = DataLoader(
                val_dataset, batch_size=1, shuffle=False, drop_last=False
            )

        network_without_ddp.eval()

        # Perform validation without gradient calculation
        val_loss_func = nn.CrossEntropyLoss(reduction="mean")
        val_loss_func_mae = nn.L1Loss(reduction="mean")
        val_losses = []
        gt_one_episode = []
        model_output_one_episode = []
        # Loop through the validation dataset
        for idx, (obs, action) in tqdm(
            enumerate(val_dataloader),
            desc="validation",
            total=len(val_dataset) // self.args["world_size"],
        ):
            # Initialize network state
            network_state = batched_space_sampler(
                network_without_ddp._state_space, batch_size=1
            )
            network_state = np_to_tensor(network_state)

            # Reset network state
            for k, v in network_state.items():
                network_state[k] = torch.zeros_like(v)

            action_predictions_logits = []
            output_actions = []

            # Infer actions for each timestep
            if self.args["using_proprioception"]:
                obs = self.calc_fk(obs)
            for i_ts in range(self.args["time_sequence_length"]):
                ob = utils.retrieve_single_timestep(obs, i_ts)
                output_action, network_state = network_without_ddp(
                    utils.dict_to_device(ob, self.device),
                    utils.dict_to_device(network_state, self.device),
                )
                output_actions.append(output_action)
                action_predictions_logits.append(
                    network_without_ddp._aux_info["action_predictions_logits"]
                )

            # Get ground truth actions
            gt_actions = network_without_ddp._action_tokenizer.tokenize(action)

            # Process predictions and ground truth actions

            # since when calculating cross entrophy, the class probability needs to be at the second dimension
            # we move the class probability from the last dimension to second dimension
            action_predictions_logits = (
                torch.cat(action_predictions_logits, dim=0)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            gt_one_episode.append(gt_actions)
            model_output_one_episode.append(action_predictions_logits.argmax(1))

            # Handle end of episode scenario
            if gt_actions[0, -1, 0] == 2:
                # gt_actions[0, -1, 0] is the terminate signal for current episode, 2 indicates the end of episode
                # whtn terminate signal is triggered, we write this episode's test results into files
                gt_one_episode = torch.cat(gt_one_episode).cpu().data.numpy()
                model_output_one_episode = (
                    torch.cat(model_output_one_episode).cpu().data.numpy()
                )

                # Visualize and store episode results
                if utils.is_main_process():
                    if not os.path.isdir(os.path.join(self.checkpoint_dir, "pics")):
                        os.mkdir(os.path.join(self.checkpoint_dir, "pics"))
                    if not os.path.isdir(
                        os.path.join(self.checkpoint_dir, "val_results")
                    ):
                        os.mkdir(os.path.join(self.checkpoint_dir, "val_results"))
                fn = (
                    "epoch_"
                    + str(epoch)
                    + "_step_"
                    + str(idx)
                    + "_gpu_"
                    + str(self.args["gpu"])
                    + ".pdf"
                )
                fn = os.path.join(self.checkpoint_dir, "val_results", fn)
                utils.visualize(gt_one_episode, model_output_one_episode, fn)
                print("result written into: ", fn)
                gt_one_episode = []
                model_output_one_episode = []

            # Calculate validation loss metrics
            val_loss = (
                val_loss_func(action_predictions_logits, gt_actions.to(self.device))
                .cpu()
                .data.numpy()
            )
            val_loss_mae = val_loss_func_mae(
                action_predictions_logits.argmax(1).float(), gt_actions.to(self.device)
            ).cpu()
            val_losses.append(val_loss)

            # Log validation metrics
            if utils.is_main_process():
                self.writer_val.add_scalar(
                    tag="loss_ce",
                    global_step=self.val_step,
                    scalar_value=val_loss,
                    walltime=time.time(),
                )
                self.writer_val.add_scalar(
                    tag="loss_mae",
                    global_step=self.val_step,
                    scalar_value=val_loss_mae.data.numpy(),
                    walltime=time.time(),
                )
                self.writer_val.add_scalar(
                    tag="epoch",
                    global_step=self.val_step,
                    scalar_value=epoch,
                    walltime=time.time(),
                )
            self.val_step += 1

        # Close the writer and return validation losses
        self.writer_val.close()
        return val_losses

    def multi_test_in_sim_env(self, epoch, network, optimizer, scheduler):
        pass

    def evaluate(self):
        val_epoch = self.args["resume_from_checkpoint"].split("/")[-1].split("-")[0]

        val_checkpoint_dir = "/".join(
            self.args["resume_from_checkpoint"].split("/")[:-1]
        )
        with open(
            os.path.join(val_checkpoint_dir, val_epoch + "_val_episodes.txt"), "w"
        ) as f:
            for val_episode_dir in self.val_dataset._episode_dirs:
                f.write(val_episode_dir + "\n")
        with open(os.path.join(val_checkpoint_dir, val_epoch + ".txt"), "w") as f:
            pass
        cam_views = ""
        for i, cmv in enumerate(self.args["cam_view"]):
            cam_views += cmv
            if i != (len(self.args["cam_view"]) - 1):
                cam_views += "_"
        subprocess.call(
            [
                "bash",
                "multi.sh",
                val_epoch,
                val_checkpoint_dir,
                self.args["resume_from_checkpoint"].split("/")[-1],
                str(self.args["gpu"]),
                cam_views,
                str(int(self.args["using_proprioception"])),
                str(self.args["num_eval_threads"]),
            ]
        )
        if self.args["distributed"]:
            torch.distributed.barrier()
        if utils.is_main_process():
            complete_rate_grab, complete_rate_lift = utils.calculate_completion_rate(
                os.path.join(val_checkpoint_dir, val_epoch + ".txt")
            )
            print("complete_rate_grab:", complete_rate_grab)
            print("complete_rate_lift:", complete_rate_lift)
            utils.merge_video(val_epoch, os.path.join(val_checkpoint_dir, "vids"))

    def make_log_dir(self, log_dir):
        """
        making the log directory
        the file structure of log dir should be:
            [log_dir]
                [log_0]
                [log_1]
                ...
                [tensorboard_logs]
                    [log_0]
                    [log_1]
                    ...
        Parameters:
        - log_dir(str): root directory storing all the logs
        Returns:
        - checkpoint_dir(str): log directory for this sepcific training
        - checkpoint_dir(str): tensorboard_dir directory for this sepcific training
        """

        id = str(time.time()).split(".")[0]
        train_name = id
        self.train_name = train_name
        if not os.path.isdir(os.path.join(log_dir)):
            os.mkdir(os.path.join(log_dir))
        checkpoint_dir = os.path.join(log_dir, train_name)
        if not os.path.isdir(os.path.join(log_dir, "tensorboard_logs")):
            os.mkdir(os.path.join(log_dir, "tensorboard_logs"))
        tensorboard_dir = os.path.join(log_dir, "tensorboard_logs", train_name)
        if utils.is_main_process():
            os.mkdir(checkpoint_dir)
        return checkpoint_dir, tensorboard_dir


if __name__ == "__main__":
    args = load_config_from_json("train_config.json")
    trainer = Trainer(args)
    if args["mode"] == "train":
        trainer.train()
    elif args["mode"] == "eval":
        trainer.evaluate()
    else:
        raise NotImplementedError("mode must be '''train''' or '''eval'''")
