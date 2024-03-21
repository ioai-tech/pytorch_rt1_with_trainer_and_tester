# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import copy
import random
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from packaging import version
from typing import Optional, List
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

if version.parse(torchvision.__version__) < version.parse("0.7"):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def make_log_dir(log_dir):
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
    if not os.path.isdir(os.path.join(log_dir)):
        os.mkdir(os.path.join(log_dir))
    checkpoint_dir = os.path.join(log_dir, train_name)
    if not os.path.isdir(os.path.join(log_dir, "tensorboard_logs")):
        os.mkdir(os.path.join(log_dir, "tensorboard_logs"))
    tensorboard_dir = os.path.join(log_dir, "tensorboard_logs", train_name)
    if is_main_process():
        os.mkdir(checkpoint_dir)
    return checkpoint_dir, tensorboard_dir, train_name


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    # 把整个batch的图片和label concat到一起  shape不一样
    # batch[0]图片  batch[1] label
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args["distributed"] = True
        args["gpu"] = int(os.environ["LOCAL_RANK"])
        args["rank"] = int(os.environ["RANK"])
        args["world_size"] = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        args["distributed"] = True
        args["gpu"] = args["rank"] % torch.cuda.device_count()
        args["rank"] = int(os.environ["SLURM_PROCID"])
        args["world_size"] = int(os.environ["WORLD_SIZE"])
    else:
        print("Not using distributed mode")
        args["distributed"] = False
        args["gpu"] = 0
        args["rank"] = 0
        args["world_size"] = 1
        return args

    torch.cuda.set_device(args["gpu"])
    args["dist_backend"] = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args["rank"], args["dist_url"]),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args["dist_backend"],
        init_method=args["dist_url"],
        world_size=args["world_size"],
        rank=args["rank"],
    )
    torch.distributed.barrier()
    return args


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse("0.7"):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )


def generate_random_color():
    """
    Generate a random color string suitable for input to Matplotlib.
    Returns:
        str: A random color string in the format '#RRGGBB'.
    """
    r = random.randint(0, 255)  # Generate a random value for the red channel
    g = random.randint(0, 255)  # Generate a random value for the green channel
    b = random.randint(0, 255)  # Generate a random value for the blue channel

    # Convert RGB channel values to hexadecimal and ensure they are always two digits
    color_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)

    return color_hex


@torch.no_grad()
def visualize(all_gt, all_output, fn):
    all_output = all_output[:, -1, :]
    all_gt = all_gt[:, -1, :]
    title = [
        "terminate_episode_l1_error: ",
        "cmd_pos_x_l1_error: ",
        "cmd_pos_y_l1_error: ",
        "cmd_pos_z_l1_error: ",
        "cmd_rot_x_l1_error: ",
        "cmd_rot_y_l1_error: ",
        "cmd_rot_z_l1_error: ",
        "cmd_gripper_l1_error: ",
    ]
    plt.figure(figsize=(22, 12))
    for i in range(8):
        c = generate_random_color()
        plt.subplot(2, 4, i + 1)
        val_loss = F.l1_loss(
            torch.from_numpy(all_output[:, i]).float(),
            torch.from_numpy(all_gt[:, i]).float(),
        )
        plt.title(title[i] + str(val_loss.cpu().data.numpy()))
        plt.plot(all_gt[:, i], c=c, label="gt")
        plt.plot(all_output[:, i], c=c, linestyle="dashed", label="output")
        plt.xlabel("timesteps")
        plt.ylabel("action_tokens")
        plt.grid()
        plt.legend()
    plt.savefig(fn, format="pdf")
    plt.clf()
    plt.close()


def retrieve_single_timestep(dict_obj, idx):
    """
    get all the values in the [dict_obj] at index [idx]
    v[:, idx], all the values in the dictionary at second dimension needs to be same
    """
    dict_obj_return = copy.deepcopy(dict_obj)
    for k, v in dict_obj.items():
        dict_obj_return[k] = v[:, idx]
    return dict_obj_return


def dict_to_device(dict_obj, device):
    """
    put all the values in the [dict_obj] to [device]
    """
    for k, v in dict_obj.items():
        assert isinstance(v, torch.Tensor)
        dict_obj[k] = v.to(device)
    return dict_obj


def set_seed(seed=3407):
    """
    set random seed to reproduce results
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_completion_rate(file_path):
    """
    Calculates the completion rates for two tasks based on a specified file.

    Parameters:
    - file_path (str): Path to the file containing task completion information.

    Returns:
    - completion_rate_0 (float): Completion rate for task_0.
    - completion_rate_1 (float): Completion rate for task_1.
    """

    # Initialize variables to count completed and total tasks for task_0 and task_1
    task_0_completed = 0
    task_0_total = 0
    task_1_completed = 0
    task_1_total = 0

    # Open the specified file for reading
    with open(file_path, "r") as file:
        # Loop through each line in the file
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            # Check the first character of the line for task_0 completion status
            if line[0] in ["0", "1"]:
                task_0_total += 1  # Increment total task_0 count
                if line[0] == "1":
                    task_0_completed += 1  # Increment completed task_0 count
            # Check the second character of the line for task_1 completion status
            if line[1] in ["0", "1"]:
                task_1_total += 1  # Increment total task_1 count
                if line[1] == "1":
                    task_1_completed += 1  # Increment completed task_1 count

    # Calculate completion rates for task_0 and task_1
    if task_1_total == 0 or task_0_total == 0:
        completion_rate_0 = 0.0  # Set completion rate to 0 if total count is 0
        completion_rate_1 = 0.0  # Set completion rate to 0 if total count is 0
    else:
        completion_rate_0 = (
            task_0_completed / task_0_total
        )  # Calculate completion rate for task_0
        completion_rate_1 = (
            task_1_completed / task_1_total
        )  # Calculate completion rate for task_1

    # Return the completion rates for task_0 and task_1
    return completion_rate_0, completion_rate_1


def merge_video(epoch, vid_dir):
    fns = os.listdir(vid_dir)
    for fn in fns:
        if not fn.endswith("mp4"):
            fns.remove(fn)
        elif not fn.startswith("e" + str(epoch) + "_"):
            fns.remove(fn)
    fns.sort()
    vid_fn_list = os.path.join(vid_dir, "videos.txt")
    with open(os.path.join(vid_fn_list), "w+") as f:
        lines = []
        for fn in fns:
            if fn.endswith("mp4"):
                lines.append("""file '""" + fn + """'""" + "\n")
        f.writelines(lines)
        f.close()
    # os.system('cd ' + dir)
    merged_vid_fn = os.path.join(vid_dir, str(epoch) + "merged.mp4")
    fast_vid_fn = os.path.join(vid_dir, str(epoch) + "fast.mp4")
    os.system(
        "ffmpeg -f concat -safe 0 -i " + vid_fn_list + " -c copy " + merged_vid_fn
    )
    time.sleep(1)
    os.system(
        "ffmpeg -i " + merged_vid_fn + ' -an -filter:v "setpts=0.1*PTS" ' + fast_vid_fn
    )
    time.sleep(1)
    print("************* remove: ", vid_fn_list, " *************")
    os.remove(vid_fn_list)
    for fn in fns:
        print("************* remove: ", os.path.join(vid_dir, fn), " *************")
        os.remove(os.path.join(vid_dir, fn))
