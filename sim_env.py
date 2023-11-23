import time
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
import glob
import pybullet as p
import threading
from panda_robot import Panda
from PIL import Image

# Dictionary defining task names and their corresponding environment classes
TASKS = {
    "touch": "TouchTaskEnv",  # Task 'touch' is associated with 'TouchTaskEnv' class
    "pick": "PickTaskEnv",  # Task 'pick' is associated with 'PickTaskEnv' class
}

# Dictionary defining various camera views along with their configuration parameters
CAM_INFO = {
    "front": [
        [0, 0, 0.7],
        1.8,
        180,
        -20,
        0,
        40,
    ],  # Front view: [position], distance, angles, fov
    "fronttop": [
        [0, 0.5, 0.7],
        1.5,
        180,
        -60,
        0,
        35,
    ],  # Front-top view: [position], distance, angles, fov
    "topdown": [
        [0, 0.35, 0],
        2.0,
        0,
        -90,
        0,
        45,
    ],  # Top-down view: [position], distance, angles, fov
    "side": [
        [0, 0.35, 0.9],
        1.5,
        90,
        0,
        0,
        40,
    ],  # Side view: [position], distance, angles, fov
    "root": [
        [0, 0.6, 0.75],
        1.3,
        -35,
        -5,
        0,
        40,
    ],  # Root view: [position], distance, angles, fov
    "wrist": [],  # Placeholder for the 'wrist', since wrist view goes with the end effector, so no predefined camera parameters required
}

# Tuple defining the resolution of the camera (width x height)
cam_resolution = (1080, 864)


def get_cam_projection_matrix(cam_view):
    """
    Calculates the camera projection matrix based on the given camera view.

    Parameters:
    - cam_view (str): Specifies the camera view.

    Returns:
    - cam_projection_matrix (list): Projection matrix for the specified camera view.
    """

    # Calculate the aspect ratio based on camera resolution
    aspect = float(cam_resolution[0]) / cam_resolution[1]
    nearVal = 0.1  # Default near clipping plane value
    farVal = 100  # Default far clipping plane value

    if cam_view == "wrist":
        # Adjust parameters for wrist camera view
        fov = 100  # Field of view for wrist camera
        nearVal = 0.018  # Adjusted near clipping plane value for wrist camera
    else:
        # Use field of view based on the specified camera view
        fov = CAM_INFO[cam_view][-1]  # Get field of view for the specified camera view

    # Compute the camera projection matrix using PyBullet's function
    cam_projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearVal,
        farVal=farVal,
    )

    # Return the calculated camera projection matrix
    return cam_projection_matrix


def get_view_matrix(cam_view, robot_id, ee_index):
    """
    Generates the view matrix for a specified camera view relative to a robot's end-effector.

    Parameters:
    - cam_view (str): Specifies the camera view.
    - robot_id (int): Identifier for the robot.
    - ee_index (int): Index of the end-effector on the robot.

    Returns:
    - cam_view_matrix (list): View matrix for the specified camera view.
    """

    if cam_view == "wrist":
        # Calculate view matrix for wrist camera view
        eye_pos, eye_ori = p.getLinkState(
            robot_id,
            ee_index,
            computeForwardKinematics=True,
        )[0:2]
        eye_pos = list(eye_pos)
        eye_pos = p.multiplyTransforms(eye_pos, eye_ori, [0, 0, -0.05], [0, 0, 0, 1])[0]
        r_mat = p.getMatrixFromQuaternion(eye_ori)
        tx_vec = np.array([r_mat[0], r_mat[3], r_mat[6]])
        ty_vec = np.array([r_mat[1], r_mat[4], r_mat[7]])
        tz_vec = np.array([r_mat[2], r_mat[5], r_mat[8]])
        camera_position = np.array(eye_pos)
        target_position = eye_pos + 0.001 * tz_vec

        # Compute view matrix for wrist camera using PyBullet's function
        cam_view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=target_position,
            cameraUpVector=ty_vec,
        )
    else:
        # Calculate view matrix for non-wrist camera views using yaw, pitch, and roll
        cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(
            CAM_INFO[cam_view][0],
            CAM_INFO[cam_view][1],
            CAM_INFO[cam_view][2],
            CAM_INFO[cam_view][3],
            CAM_INFO[cam_view][4],
            2,
        )

    # Return the computed camera view matrix
    return cam_view_matrix


def get_cam_view_img(cam_view, robot_id=None, ee_index=None):
    """
    Captures an image from a specified camera view using PyBullet.

    Parameters:
    - cam_view (str): Specifies the camera view.
    - robot_id (int, optional): Identifier for the robot.
    - ee_index (int, optional): Index of the end-effector on the robot.

    Returns:
    - img (numpy.ndarray): Captured image from the specified camera view.
    """

    # Obtain the view matrix for the camera view
    cam_view_matrix = get_view_matrix(cam_view, robot_id, ee_index)

    # Obtain the projection matrix for the camera view
    cam_projection_matrix = get_cam_projection_matrix(cam_view)

    # Capture the camera image using PyBullet
    (width, height, rgb_pixels, _, _) = p.getCameraImage(
        cam_resolution[0],
        cam_resolution[1],
        viewMatrix=cam_view_matrix,
        projectionMatrix=cam_projection_matrix,
    )

    # Reshape and process the image data
    rgb_array = np.array(rgb_pixels).reshape((height, width, 4)).astype(np.uint8)
    img = np.array(resize_and_crop(rgb_array[:, :, :3]))  # Process the image

    # Return the captured and processed image
    return img


def resize_and_crop(input_image):
    """
    Crop the image to a 5:4 aspect ratio and resize it to 320x256 pixels.

    Parameters:
    - input_image (numpy.ndarray): Input image data in array format.

    Returns:
    - input_image (PIL.Image.Image): Cropped and resized image in PIL Image format.
    """

    # Convert the input image array to a PIL Image
    input_image = Image.fromarray(input_image)

    # Get the width and height of the input image
    width, height = input_image.size

    # Define target and current aspect ratios
    target_aspect = 5 / 4
    current_aspect = width / height

    if current_aspect > target_aspect:
        # If the image is too wide, crop its width
        new_width = int(target_aspect * height)
        left_margin = (width - new_width) / 2
        input_image = input_image.crop((left_margin, 0, width - left_margin, height))
    elif current_aspect < target_aspect:
        # If the image is too tall, crop its height
        new_height = int(width / target_aspect)
        top_margin = (height - new_height) / 2
        input_image = input_image.crop((0, top_margin, width, height - top_margin))

    # Resize the cropped image to 320x256 pixels
    input_image = input_image.resize((320, 256))

    # Return the cropped and resized image as a PIL Image
    return input_image


class SimEnv(object):
    def __init__(self):
        # Set solverResidualThreshold to 0 for physics engine parameter
        p.setPhysicsEngineParameter(solverResidualThreshold=0)

        # Control time step and reset environment wait time
        self.control_dt = 1.0 / 240.0
        self.reset_env_wait_time = 0.5

        # Initialize attributes related to the robot, target object, poses, state, waypoints, and data recording
        self.robot = None
        self.tar_obj = None
        self.tar_obj_pose = None
        self.state = None
        self.target_waypoints = None
        self.data_record_fq = None
        self.collected_traj = 700

        # Load environment, set up camera, and reset environment
        self.load_env()
        self.set_camera()
        self.reset_env()

        # Initialize a lock for thread safety
        self.lock = threading.Lock()

    def load_env(self):
        raise NotImplementedError

    def reset_env(self):
        raise NotImplementedError

    def set_camera(self):
        # Set camera resolution
        self.cam_resolution = (1080, 864)  # Define camera resolution

        # Retrieve camera information from CAM_INFO dictionary
        self.cam_info = CAM_INFO

        # Initialize an empty list to store view matrices for each camera view
        self.cam_view_matrices = []

        # Iterate through each camera view in CAM_INFO and compute view matrices
        for key, val in self.cam_info.items():
            if key == "wrist":
                self.cam_view_matrices.append([])  # Placeholder for 'wrist' view
            else:
                # Compute view matrix using yaw, pitch, roll, and other parameters from CAM_INFO
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    val[0], val[1], val[2], val[3], val[4], 2
                )
                self.cam_view_matrices.append(
                    view_matrix
                )  # Store the computed view matrix

        # Compute projection matrix for the camera with specified FOV, aspect ratio, and depth range
        self.cam_projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.cam_resolution[0]) / self.cam_resolution[1],
            nearVal=0.1,
            farVal=100,
        )


class TouchTaskEnv(SimEnv):
    def __init__(self):
        super().__init__()

    def load_env(self):
        p.loadURDF("table/table.urdf", [0, 0.35, 0], [0, 0, 0, 1])
        self.tar_obj = p.loadURDF("urdf/cube/cube.urdf", [0, 0, 0], globalScaling=0.04)
        p.changeVisualShape(self.tar_obj, -1, rgbaColor=[1, 0, 0, 1])
        self.robot = Panda()
        self.robot.load()

    def reset_tar_obj(self, tar_obj_range=None, tar_pos_rot=None, random_pos_rot=True):
        if random_pos_rot:
            # Generate random position and rotation within the specified range
            x = random.uniform(tar_obj_range[0], tar_obj_range[1])
            y = random.uniform(tar_obj_range[2], tar_obj_range[3])
            r = random.uniform(tar_obj_range[4], tar_obj_range[5])
            pos = [x, y, 0.645]  # Set the z-coordinate (height) of the object
            rot = p.getQuaternionFromEuler(
                [0, np.pi / 2, r]
            )  # Convert Euler angles to quaternion
        else:
            # Use provided target position and rotation
            x, y, r = tar_pos_rot[0], tar_pos_rot[1], tar_pos_rot[4]
            pos = [x, y, 0.645]
            rot = [tar_pos_rot[2], tar_pos_rot[3], tar_pos_rot[4], tar_pos_rot[5]]

        # Reset the target object's position and orientation
        p.resetBasePositionAndOrientation(
            self.tar_obj,
            pos,
            rot,
        )

        # Update the stored target object's pose (position and orientation)
        self.tar_obj_pose = p.getBasePositionAndOrientation(self.tar_obj)

    def reset_env(self):
        # Reset the robot's joints to their home positions
        self.robot.reset_j_home()

        # Pause execution for 1 second to allow for resetting
        time.sleep(1)

        # Reset state variables related to the environment
        self.state = 0  # Reset the state to 0
        self.t = 0  # Reset time counter to 0
        self.state_stuck_t = 0  # Reset the state stuck time to 0


class PickTaskEnv(TouchTaskEnv):
    def __init__(self):
        super().__init__()
