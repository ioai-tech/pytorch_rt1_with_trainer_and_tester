import copy
import os
import warnings

warnings.filterwarnings("ignore")


# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import time
import subprocess
import random
import json
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
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

# from dataset import build_dataset
from maruya24_rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from maruya24_rt1.transformer_network import TransformerNetwork
from maruya24_rt1.tokenizers.action_tokenizer import RT1ActionTokenizer
from maruya24_rt1.transformer_network_test_set_up import state_space_list
from luciRT1 import MaxViT, RT1


def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


class Trainer:
    def __init__(self, args):
        utils.set_seed()
        self.args = args
        self.args = utils.init_distributed_mode(self.args)
        self.checkpoint_dir, self.tensorboard_dir, self.train_name = utils.make_log_dir(
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
                        spaces.Box(
                            low=-0.015, high=0.015, shape=(3,), dtype=np.float32
                        ),
                    ),
                    (
                        "rotation_delta",
                        spaces.Box(
                            low=-np.pi / 100,
                            high=np.pi / 100,
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
        # network = TransformerNetwork(**network_configs)

        vit = MaxViT(
            num_classes=1000,
            dim_conv_stem=64,
            dim=96,
            dim_head=32,
            depth=(2, 2, 5, 2),
            window_size=7,
            mbconv_expansion_rate=4,
            mbconv_shrinkage_rate=0.25,
            dropout=0.1,
        )

        network = RT1(
            vit=vit, num_actions=8, depth=6, heads=8, dim_head=64, cond_drop_prob=0.2
        )

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
        action_tokenizer = RT1ActionTokenizer(
            self._action_space, vocab_size=256  # action space
        )
        epoch_start = checkpoint["epoch"] if self.args["resume"] else 0
        for e in range(epoch_start, self.args["epochs"]):
            network.train()
            with tqdm(
                train_dataloader, dynamic_ncols=True, desc="train"
            ) as tqdmDataLoader:
                for _, (obs, action) in enumerate(tqdmDataLoader):
                    # Perform training steps
                    optimizer.zero_grad()
                    action = action_tokenizer.tokenize(action).flatten(0, 2).cuda()
                    img = obs["image"].cuda()
                    lang = obs["natural_language_embedding"]
                    # if self.args["using_proprioception"]:
                    #     obs = self.calc_fk(obs)
                    # obs = utils.dict_to_device(obs, self.device)
                    # network_state = utils.dict_to_device(network_state, self.device)
                    output_actions = network(img, lang).flatten(0, 2)

                    loss = F.cross_entropy(output_actions, action)

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
                                round(torch.cuda.max_memory_allocated() / (1024**3), 2)
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
                    "model_state_dict": (
                        network_without_ddp.state_dict()
                        if self.args["distributed"]
                        else network.state_dict()
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "action_space": self._action_space,
                    "epoch": e,
                }
                utils.save_on_master(checkpoint, checkpoint_filename)
            scheduler.step()

    def test_in_sim_env(self, epoch, network, optimizer, scheduler):
        pass


if __name__ == "__main__":
    args = load_config_from_json("train_config.json")
    trainer = Trainer(args)
    if args["mode"] == "train":
        trainer.train()
    elif args["mode"] == "eval":
        trainer.evaluate()
    else:
        raise NotImplementedError("mode must be '''train''' or '''eval'''")
