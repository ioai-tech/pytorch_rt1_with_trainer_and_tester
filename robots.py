import pybullet as p
import numpy as np
import random
import pybullet_data as pdata
from utils import debug_draw_pose
import math
from robot_descriptions.loaders.pybullet import load_robot_description


class RobotBase(object):
    def __init__(self) -> None:
        self.arm_dof = None
        self.ee_index = None
        self.finger_index = None
        self.home_j_pos = None
        self.gripper_range = None
        self.is_hand = False
        self.home_ee_pose = ((-0.0395, 0.415, 1.145), (1, 0, 0, 0))
        self.grasp_force_threshold = None

    def load(self, urdf_path, base_pos, base_ori):
        self.robot_id = p.loadURDF(
            urdf_path,
            base_pos,
            base_ori,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )

    def calc_ik(self, pose):
        return list(
            p.calculateInverseKinematics(
                self.robot_id,
                self.ee_index,
                pose[0],
                pose[1],
                [-7] * 7,
                [7] * 7,
                [7] * 7,
                self.home_j_pos,
                maxNumIterations=100,
                solver=p.IK_DLS,
            )
        )

    def reset_j_home(self, random_home=False):
        if random_home:
            for i in range(self.arm_dof):
                self.home_j_pos[i] += random.uniform(-np.pi / 10, np.pi / 10)
        # self.home_j_pos = [1.22, -0.458, 0.31, -3.0, 0.20, 4* math.pi / 5, 0, 0.04, 0.04]
        self.reset_j_pos(self.home_j_pos)

    def reset_j_pos(self, j_pos):
        index = 0
        for j in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0)
            joint_type = p.getJointInfo(self.robot_id, j)[2]
            if joint_type in [
                p.JOINT_PRISMATIC,
                p.JOINT_REVOLUTE,
            ]:
                p.resetJointState(self.robot_id, j, j_pos[index])
                index = index + 1

    def move_arm(self, arm_pos, max_vel, force):
        for i in range(self.arm_dof):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                arm_pos[i],
                maxVelocity=max_vel[i],
                force=force[i],
            )

    def move_gripper(self, gripper_pos, max_vel, force):
        for i, j_idx in enumerate(self.finger_index):
            p.setJointMotorControl2(
                self.robot_id,
                j_idx,
                p.POSITION_CONTROL,
                gripper_pos[i],
                maxVelocity=max_vel[i],
                force=force[i],
            )

    def move_j(self):
        raise NotImplementedError

    def is_j_arrived(self, j_pos, include_finger=True, threshold=1e-2):
        cur_joint_position = [
            s[0]
            for s in p.getJointStates(
                self.robot_id, list(range(self.arm_dof)) + self.finger_index
            )
        ]
        diff_arm = np.abs(
            np.array(cur_joint_position)
            - np.array(j_pos)[: self.arm_dof + len(self.finger_index)]
        )
        is_arrive = np.all(diff_arm[: self.arm_dof - 1] <= threshold)
        # if include_finger:
        #     is_arrive = is_arrive and np.all(
        #         diff_arm[-len(self.finger_index) :] <= threshold
        #     )
        return is_arrive

    def is_ee_arrived(self, ee_pose, tar_obj_id=None, threshold=2 * 1e-2):
        robot_ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_index)[0])
        diff_pos = np.abs(robot_ee_pos - ee_pose[0])
        is_arrive = np.all(diff_pos <= threshold)
        if tar_obj_id != None:
            is_arrive = (
                is_arrive
                and p.getClosestPoints(self.robot_id, tar_obj_id, 1e-5)
                and p.getContactPoints(self.robot_id, tar_obj_id)
                and p.getContactPoints(self.robot_id, tar_obj_id)[0][9]
                > self.grasp_force_threshold
            )
            # p.getContactPoints(self.robot_id, tar_obj_id)
        return is_arrive


class UR5(RobotBase):
    def __init__(self):
        super().__init__()
        self.arm_dof = 6
        self.ee_index = 18
        self.finger_index = [8]
        self.home_j_pos = [1.40, -1.58, 0.94, -0.93, -1.57, -1.74] + [0.0] * 6
        self.gripper_range = [0, 0.085]
        self.grasp_force_threshold = 30

    def load(self):
        super().load(
            "urdf/ur5_robotiq/urdf/ur5_robotiq_85.urdf", [0, 0, 0.62], [0, 0, 0, 1]
        )
        finger_child_multiplier = {10: -1, 12: 1, 13: 1, 15: -1, 17: 1}
        for joint_id, multiplier in finger_child_multiplier.items():
            c = p.createConstraint(
                self.robot_id,
                self.finger_index[0],
                self.robot_id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1
            )  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, gripper_state):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_length = (
            gripper_state * (self.gripper_range[1] - self.gripper_range[0])
            + self.gripper_range[0]
        )
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        # angle calculation
        super().move_gripper([open_angle], max_vel=[4], force=[1000])

    def move_arm(self, j_pos):
        super().move_arm(
            j_pos[: self.arm_dof],
            [1] * self.arm_dof,
            [5 * 240.0] * self.arm_dof,
        )


class Panda(RobotBase):
    def __init__(self):
        super().__init__()
        self.arm_dof = 7
        self.ee_index = 11
        self.finger_index = [9, 10]
        self.home_j_pos = [1.22, -0.458, 0.31, -2.0, 0.20, 1.56, 2.32, 0.04, 0.04]
        self.gripper_range = [0.01, 0.04]
        self.grasp_force_threshold = 2

    def load(self):
        super().load("franka_panda/panda.urdf", [0, 0, 0.62], [0, 0, 0, 1])
        # create a constraint to keep the fingers centered, 9 and 10 for finger indices
        c = p.createConstraint(
            self.robot_id,
            self.finger_index[0],
            self.robot_id,
            self.finger_index[1],
            jointType=p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    def move_arm(self, j_pos):
        super().move_arm(
            j_pos[: self.arm_dof],
            [1 if i == self.arm_dof - 1 else 0.5 for i in range(self.arm_dof)],
            [5 * 240.0] * self.arm_dof,
        )

    def move_gripper(self, gripper_state):
        gripper_pos = (
            gripper_state * (self.gripper_range[1] - self.gripper_range[0])
            + self.gripper_range[0]
        )
        super().move_gripper([gripper_pos] * 2, [0.1] * 2, [1000] * 2)


class Iiwa(RobotBase):
    def __init__(self):
        super().__init__()
        self.arm_dof = 7
        self.ee_index = 14
        self.finger_index = [8, 10, 11, 13]
        self.home_j_pos = [
            -0.695,
            -0.117,
            -0.692,
            1.261,
            0.140,
            -1.773,
            0.125,
            0.125,
            -0.5,
            0.5,
            0.5,
            -0.5,
        ]

        self.gripper_range = [0, 0.5]
        self.grasp_force_threshold = 20

    def load(self):
        self.robot_id = p.loadSDF("urdf/kuka_iiwa/kuka_with_gripper2.sdf")[0]
        p.resetBasePositionAndOrientation(self.robot_id, [-0.1, 0, 0.7], [0, 0, 0, 1])

    def move_gripper(self, gripper_state):
        val = (
            gripper_state * (self.gripper_range[1] - self.gripper_range[0])
            + self.gripper_range[0]
        )
        gripper_pos = [-val, val, val, -val]
        super().move_gripper(gripper_pos, max_vel=[0.5] * 4, force=[1000] * 4)

    def move_arm(self, j_pos):
        super().move_arm(
            j_pos[: self.arm_dof],
            [1] * self.arm_dof,
            [5 * 240.0] * self.arm_dof,
        )


class Xarm(RobotBase):
    def __init__(self):
        super().__init__()
        self.arm_dof = 6
        self.ee_index = 13
        self.finger_index = [7, 8, 9, 10, 11, 12]
        # self.finger_index = [7]
        self.home_j_pos = [1.623, -0.0326, -1.939, 0.070, 1.960, 1.683] + [0] * 6
        self.gripper_range = [0, 0.85]
        self.grasp_force_threshold = 10

    def load(self):
        super().load("urdf/xarm/xarm6_with_gripper.urdf", [0, 0, 0.62], [0, 0, 0, 1])

    def move_gripper(self, gripper_state):
        val = (1 - gripper_state) * (
            self.gripper_range[1] - self.gripper_range[0]
        ) + self.gripper_range[0]
        gripper_pos = [val] * len(self.finger_index)
        super().move_gripper(
            gripper_pos,
            max_vel=[1] * len(self.finger_index),
            force=[100000] * len(self.finger_index),
        )

    def move_arm(self, j_pos):
        super().move_arm(
            j_pos[: self.arm_dof],
            [1] * self.arm_dof,
            [5 * 240.0] * self.arm_dof,
        )


class HandBase(RobotBase):
    def __init__(self):
        super().__init__()

    def move_ee(self, ee_pose):
        cur_ee_pose = p.getLinkState(self.robot_id, self.ee_index)[0:2]
        inv_cur_ee_pose = p.invertTransform(cur_ee_pose[0], cur_ee_pose[1])
        delta_pose = p.multiplyTransforms(
            inv_cur_ee_pose[0], inv_cur_ee_pose[1], ee_pose[0], ee_pose[1]
        )
        delta_pos = delta_pose[0]
        delta_rot = p.getEulerFromQuaternion(delta_pose[1])
        max_delta_pos = 0.01
        max_delta_rot = np.pi / 50

        delta_pos = np.clip(delta_pos, -max_delta_pos, max_delta_pos)
        delta_rot = np.clip(delta_rot, -max_delta_rot, max_delta_rot)
        ee_pose = p.multiplyTransforms(
            cur_ee_pose[0],
            cur_ee_pose[1],
            delta_pos,
            p.getQuaternionFromEuler(delta_rot),
        )
        p.changeConstraint(self.hand_cid, ee_pose[0], ee_pose[1], maxForce=100, erp=10)

    def move_gripper(self, gripper_status):
        for index, range in self.gripper_range.items():
            val = ((1 - gripper_status) * (range[1] - range[0]) + range[0]) * range[2]
            p.setJointMotorControl2(
                self.robot_id,
                index,
                p.POSITION_CONTROL,
                val,
                maxVelocity=8,
                force=100,
            )

    def is_ee_arrived(self, ee_pose, tar_obj_id=None, threshold=2 * 1e-2):
        robot_ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_index)[0])
        diff_pos = np.abs(robot_ee_pos - ee_pose[0])
        is_arrive = np.all(diff_pos <= threshold)
        if tar_obj_id != None:
            for i, (key, val) in enumerate(self.gripper_range.items()):
                is_arrive = is_arrive and np.all(
                    abs(p.getJointState(self.robot_id, key)[0] - val[1] * val[2]) <= 0.2
                )
        return is_arrive


class MplHand(HandBase):
    def __init__(self):
        super().__init__()
        self.ee_index = 7
        self.arm_dof = 3
        self.home_j_pos = [0] * 22
        self.is_hand = True
        self.finger_index = [8, 10, 12, 18, 20]
        self.gripper_range = {
            8: [0, 1.8, 1],
            10: [0, 1, -1],
            12: [0, 2.0, 1],
            18: [0, 1.57, 1],
            20: [0, 0.5, 1],
        }
        self.grasp_force_threshold = 2

    def load(self):
        self.robot_id = p.loadMJCF("urdf/MPL/MPL.xml")[0]
        p.resetBasePositionAndOrientation(
            self.robot_id, self.home_ee_pose[0], (0, 0, 1, 0)
        )
        for i in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 1.05, 0)
        for i in range(22):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 0, 0)
        self.hand_cid = p.createConstraint(
            self.robot_id,
            self.ee_index,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        )
        state = random.getstate()
        random.seed(6)
        link_indeices = [-1, 1, 6] + list(range(9, 24, 2)) + list(range(24, 47, 2))
        for i in link_indeices:
            p.changeVisualShape(
                self.robot_id,
                i,
                rgbaColor=[
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    random.uniform(0, 1),
                    1,
                ],
            )
        random.setstate(state)
        p.changeConstraint(self.hand_cid, self.home_ee_pose[0], (1, 0, 0, 0))


class AllegroHand(HandBase):
    def __init__(self):
        super().__init__()
        self.ee_index = 21
        self.arm_dof = 3
        self.home_j_pos = [0] * 22
        self.is_hand = True
        self.finger_index = [2, 3, 16, 18]
        self.gripper_range = {
            2: [0, 1.8, 1],
            3: [0, 0.2, 1],
            16: [0, 1.4, 1],
            18: [0, 0.2, 1],
        }
        self.grasp_force_threshold = 2

    def load(self):
        self.robot_id = p.loadURDF(
            "urdf/allegro_hand_description/urdf/allegro_hand_description_right.urdf"
        )
        p.resetBasePositionAndOrientation(
            self.robot_id, self.home_ee_pose[0], [0, 0.707, 0, 0.707]
        )
        for i in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, 0, 0)
        self.hand_cid = p.createConstraint(
            self.robot_id,
            self.ee_index,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        )
        p.changeConstraint(self.hand_cid, self.home_ee_pose[0], [1, 0, 0, 0])


if __name__ == "__main__":
    pass
