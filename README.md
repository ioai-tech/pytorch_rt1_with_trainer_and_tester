# Reproducing RT-1 in PyTorch

[![IO](https://img.shields.io/badge/io%20intelligence%20-000000)](https://io-ai.tech)

This repository contains complete PyTorch implementation for **RT-1** based on the paper: [RT-1 Paper](https://arxiv.org/abs/2212.06817) and implementation of RT-1 model by **maruya24's RT-1**: [maruya24's RT-1 GitHub](https://github.com/maruya24/pytorch_robotics_transformer). Our implementation complete the training, validation pipeline, and soon-to-come evaluation pipeline.

## Acknowledgements

- **maruya24**: For their work on RT-1, which serves as the foundation for this implementation - [maruya24's RT-1 GitHub](https://github.com/maruya24/pytorch_robotics_transformer)
    - changes on model structure: similar to what it looks like in [diffusion policy](https://diffusion-policy.cs.columbia.edu/), robot's end effector's position and orientation is concatenated to the end of the sequence before it is sent into the transformer
- **detr (Facebook)**: Utilities for distributed training from DETR - [detr/util/misc.py](https://github.com/facebookresearch/detr/blob/main/util/misc.py).

## Training

To train RT-1 in distributed mode with 4 GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=[your_gpu_num] --use_env IO_trainer_torch.py
```

The training configuration is stored in train_config.json.

## Training Configuration

The training configuration for RT-1 includes various parameters that influence the training process. Here's an overview of some key configurations:

- **Mode:** Training mode, options ['train', 'eval']
- **Device:** CUDA device for computation, options ['cpu', 'cuda']
- **Data Path:** Path to the dataset/[robotname]_[taskname]
- **Camera Views:** Views used in training (`front`, `wrist`, ... see these in dataset folder)
- **Log Directory:** Directory to store logs
- **Time Sequence Length:** Length of the time sequence (e.g., 6), RT-1 takes history timesteps of images as part of model input, which means `1` frame of current timestep image and `time_sequence_length - 1` frames of history image.
- **Learning Rate:** Initial learning rate
- **Batch Size:** Size of each training batch
- **Epochs:** Number of training epochs
- **Resume:** Whether to resume training from a checkpoint
- **Resume from checkpoint:** resume training from checkpoint path
- **World size:** how many gpus you are intended to use during training
- **Dist url:** distributed urls used for initialize distributed training (e.g., `"env://"`)
- **Validation Interval:** Interval between validation steps
- **Num train episode:** number of training episode used
- **Num val episode:** number of validation episode used
- **Network Configurations:** Parameters for the network architecture
- **Scheduler Configurations:** Parameters for the learning rate scheduler

### Example Configuration


```json
{
    "mode": "train",
    "device": "cuda",
    "data_path": "IO_pybullet_open_dataset/Panda_pick",
    "cam_view" : ["front", "wrist"],
    "log_dir": "/mnt/logs_1",
    "time_sequence_length": 6,
    "lr": 0.0001,
    "batch_size": 6,
    "epochs": 50,
    "resume": false,
    "resume_from_checkpoint": "",
    "predicting_next_ts": true,
    "world_size": 4,
    "dist_url": "env://",
    "val_interval" : 25,
    "num_val_threads": 25,
    "num_train_episode" : 200,
    "num_val_episode" : 10,
    "using_proprioception" : false,
    "network_configs": {
        "vocab_size" : 256,
        "token_embedding_size_per_image" : 512,
        "language_embedding_size" : 512,
        "num_layers" : 8,
        "layer_size" : 128,
        "num_heads" : 8,
        "feed_forward_size" : 512,
        "dropout_rate" : 0.1,
        "crop_size" : 236,
        "use_token_learner" : true
    },
    "scheduler_configs" : {
        "T_0" : 10,
        "T_mult" : 2,
        "eta_min" : 1e-6,
        "verbose" : true
    }
    
}

```
Pretrained weights trained on settings above can be downloaded from [pretrained_weights](https://drive.google.com/uc?export=download&id=1USLqOfqYfqIrigx1hkY37SGyrLJuaPwc), setting "resume_from_checkpoint" to path of pretrained weight and setting "resume" to True to resume from the checkpoint.


## Limitations

- We are currently validating the code in the PyBullet environment. Validation code will be added within a week.
- The mode is presently limited to single-task training.

## Dataset Structure

Our dataset follows a specific file structure:

- [robotname]_[taskname]
  - [cam_view_0]
    - data_000
      - rgb # Images
        - image_001.png
        - image_002.png
        - ...
      - results.csv # Robot actions
      - results_raw.csv # Joint and target object positions
    - data_001
    - ...
  - [cam_view_1]
    - data_000
    - data_001
    - ...
  - ...

Simliar to Robomimic's lift mission [robomimic](https://robomimic.github.io/), we collected dataset from third-person and first-person perspectives.

You can download our dataset collected from PyBullet [IO_open_dataset](https://drive.google.com/uc?export=download&id=1RoTxnipQf2SIXqzvroDOXAleNvvNIIwZ).

## Contacts

Join wechat group for discussion


![wechat group](img/wechatgroup.jpg)

