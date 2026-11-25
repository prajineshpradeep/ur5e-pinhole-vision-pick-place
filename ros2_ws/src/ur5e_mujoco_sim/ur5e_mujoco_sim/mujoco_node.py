"""ROS 2 Humble bridge node for the UR5e + Custom Gripper MuJoCo simulation."""

from __future__ import annotations

import array
from dataclasses import dataclass, field
import os
import sys
import threading
import time

# Allow system ROS 2 python3 to import mujoco & numpy from the project's Python 3.10 .venv
_VENV_SITE = "/home/praji/mujoco_project/.venv/lib/python3.10/site-packages"
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

# Use EGL for offscreen RGB-D Renderer so it coexists cleanly with GLFW passive viewer
os.environ.setdefault("MUJOCO_GL", "egl")

from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_JOINTS = ["left_finger_joint", "right_finger_joint"]
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS


@dataclass
class _ActiveTrajectory:
    """Thread-safe container for an active FollowJointTrajectory execution."""

    joint_names: list[str]
    ctrl_indices: list[int]
    times: np.ndarray
    positions: np.ndarray
    start_sim_time: float
    done_event: threading.Event = field(default_factory=threading.Event)
    canceled: bool = False
    error_code: int = FollowJointTrajectory.Result.SUCCESSFUL

    def sample(self, sim_time: float) -> tuple[np.ndarray, bool]:
        """Interpolate joint target positions at sim_time; return (targets, is_done)."""
        elapsed = max(0.0, sim_time - self.start_sim_time)
        if len(self.times) == 0:
            return np.array([], dtype=np.float64), True
        if elapsed >= self.times[-1]:
            return self.positions[-1].copy(), True

        idx = int(np.searchsorted(self.times, elapsed, side="right")) - 1
        idx = max(0, min(idx, len(self.times) - 2))
        t0, t1 = float(self.times[idx]), float(self.times[idx + 1])
        alpha = (elapsed - t0) / max(1e-9, t1 - t0)
        q = (1.0 - alpha) * self.positions[idx] + alpha * self.positions[idx + 1]
        return q, False


class MuJoCoSimNode(Node):
    """Steps MuJoCo physics, publishes joint states + pinhole RGB-D, and executes MoveIt trajectories."""

    def __init__(self) -> None:
        super().__init__("mujoco_sim_node")

        default_mjcf = os.path.join(
            get_package_share_directory("ur5e_mujoco_sim"), "mjcf", "ur5e_scene.xml"
        )
        default_headless = not bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        self.declare_parameter("mjcf_path", default_mjcf)
        self.declare_parameter("headless", default_headless)
        self.declare_parameter("cam_width", 640)
        self.declare_parameter("cam_height", 480)
        self.declare_parameter("cam_fps", 10.0)
        self.declare_parameter("joint_state_hz", 100.0)

        mjcf_path = self.get_parameter("mjcf_path").get_parameter_value().string_value
        self.headless = self.get_parameter("headless").get_parameter_value().bool_value
        self.cam_width = self.get_parameter("cam_width").get_parameter_value().integer_value
        self.cam_height = self.get_parameter("cam_height").get_parameter_value().integer_value
        self.cam_fps = self.get_parameter("cam_fps").get_parameter_value().double_value
        self.joint_state_hz = self.get_parameter("joint_state_hz").get_parameter_value().double_value

        self.get_logger().info(f"Loading MuJoCo scene: {mjcf_path} (headless={self.headless})")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        # Map joint & actuator indices
        self._jnt_qpos_adr: dict[str, int] = {}
        self._jnt_dof_adr: dict[str, int] = {}
        for jname in ALL_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Joint '{jname}' missing in MJCF model.")
            self._jnt_qpos_adr[jname] = int(self.model.jnt_qposadr[jid])
            self._jnt_dof_adr[jname] = int(self.model.jnt_dofadr[jid])

        self._ctrl_map: dict[str, int] = {
            "shoulder_pan_joint": 0,
            "shoulder_lift_joint": 1,
            "elbow_joint": 2,
            "wrist_1_joint": 3,
            "wrist_2_joint": 4,
            "wrist_3_joint": 5,
            "left_finger_joint": 6,
        }

        # Pinhole camera setup (eye-in-hand wrist_cam)
        self._cam_name = "wrist_cam"
        self._cam_frame_id = "wrist_cam_optical_frame"
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam_name)
        if self._cam_id < 0:
            raise RuntimeError(f"Camera '{self._cam_name}' missing in MJCF model.")
        self._camera_info_template = self._build_camera_info_template()
        self.renderer = mujoco.Renderer(
            self.model, height=self.cam_height, width=self.cam_width
        )

        # Thread synchronization & active trajectories
        self._lock = threading.Lock()
        self._active_arm_traj: _ActiveTrajectory | None = None
        self._active_gripper_traj: _ActiveTrajectory | None = None

        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.rgb_pub = self.create_publisher(Image, "/wrist_cam/image_raw", 10)
        self.depth_pub = self.create_publisher(Image, "/wrist_cam/depth/image_raw", 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, "/wrist_cam/camera_info", 10)

        # Action servers matching ur5e_moveit_config/config/moveit_controllers.yaml
        cb_group = ReentrantCallbackGroup()
        self._arm_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
            execute_callback=self._execute_arm_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=cb_group,
        )
        self._gripper_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
            execute_callback=self._execute_gripper_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=cb_group,
        )
        self.get_logger().info(
            "MuJoCo ROS 2 node ready: /arm_controller/follow_joint_trajectory & "
            "/gripper_controller/follow_joint_trajectory (eye-in-hand /wrist_cam/*)"
        )

    def _build_camera_info_template(self) -> CameraInfo:
        fovy_rad = np.deg2rad(float(self.model.cam_fovy[self._cam_id]))
        fy = self.cam_height / (2.0 * np.tan(fovy_rad / 2.0))
        fx = fy
        cx = (self.cam_width - 1) / 2.0
        cy = (self.cam_height - 1) / 2.0

        info = CameraInfo()
        info.header.frame_id = self._cam_frame_id
        info.width = int(self.cam_width)
        info.height = int(self.cam_height)
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _goal_callback(self, _goal_request: FollowJointTrajectory.Goal) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_arm_trajectory(self, goal_handle) -> FollowJointTrajectory.Result:
        return self._execute_trajectory(goal_handle, is_arm=True)

    def _execute_gripper_trajectory(self, goal_handle) -> FollowJointTrajectory.Result:
        return self._execute_trajectory(goal_handle, is_arm=False)

    def _execute_trajectory(self, goal_handle, is_arm: bool) -> FollowJointTrajectory.Result:
        traj = goal_handle.request.trajectory
        joint_names = list(traj.joint_names)
        if not traj.points:
            goal_handle.succeed()
            res = FollowJointTrajectory.Result()
            res.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return res

        try:
            ctrl_indices = [self._ctrl_map[name] for name in joint_names]
        except KeyError as exc:
            self.get_logger().error(f"Unknown joint in trajectory: {exc}")
            goal_handle.abort()
            res = FollowJointTrajectory.Result()
            res.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return res

        times = np.array(
            [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in traj.points],
            dtype=np.float64,
        )
        positions = np.array([list(p.positions) for p in traj.points], dtype=np.float64)

        with self._lock:
            active = _ActiveTrajectory(
                joint_names=joint_names,
                ctrl_indices=ctrl_indices,
                times=times,
                positions=positions,
                start_sim_time=float(self.data.time),
            )
            if is_arm:
                if self._active_arm_traj is not None:
                    self._active_arm_traj.canceled = True
                    self._active_arm_traj.done_event.set()
                self._active_arm_traj = active
            else:
                if self._active_gripper_traj is not None:
                    self._active_gripper_traj.canceled = True
                    self._active_gripper_traj.done_event.set()
                self._active_gripper_traj = active

        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = joint_names
        while not active.done_event.wait(timeout=0.05):
            if not rclpy.ok():
                active.canceled = True
                goal_handle.abort()
                res = FollowJointTrajectory.Result()
                res.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                return res
            if goal_handle.is_cancel_requested:
                with self._lock:
                    active.canceled = True
                    if is_arm and self._active_arm_traj is active:
                        self._active_arm_traj = None
                    elif not is_arm and self._active_gripper_traj is active:
                        self._active_gripper_traj = None
                goal_handle.canceled()
                res = FollowJointTrajectory.Result()
                res.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return res
            with self._lock:
                feedback.actual.positions = [
                    float(self.data.qpos[self._jnt_qpos_adr[n]]) for n in joint_names
                ]
            goal_handle.publish_feedback(feedback)

        result = FollowJointTrajectory.Result()
        if active.canceled:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
        else:
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def _step_trajectories_locked(self) -> None:
        sim_t = float(self.data.time)
        for attr in ("_active_arm_traj", "_active_gripper_traj"):
            active: _ActiveTrajectory | None = getattr(self, attr)
            if active is None:
                continue
            q_target, finished = active.sample(sim_t)
            for idx, val in zip(active.ctrl_indices, q_target):
                self.data.ctrl[idx] = float(val)
            if finished:
                active.done_event.set()
                setattr(self, attr, None)

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ALL_JOINTS
        with self._lock:
            msg.position = [float(self.data.qpos[self._jnt_qpos_adr[n]]) for n in ALL_JOINTS]
            msg.velocity = [float(self.data.qvel[self._jnt_dof_adr[n]]) for n in ALL_JOINTS]
        self.joint_state_pub.publish(msg)

    def _publish_camera_rgbd(self) -> None:
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=self._cam_name)
            rgb = self.renderer.render().copy()

            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=self._cam_name)
            depth = self.renderer.render().astype(np.float32)
            self.renderer.disable_depth_rendering()

        rgb_msg = Image()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self._cam_frame_id
        rgb_msg.height = int(self.cam_height)
        rgb_msg.width = int(self.cam_width)
        rgb_msg.encoding = "rgb8"
        rgb_msg.is_bigendian = 0
        rgb_msg.step = int(self.cam_width * 3)
        rgb_msg._data = array.array("B", rgb.tobytes())

        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self._cam_frame_id
        depth_msg.height = int(self.cam_height)
        depth_msg.width = int(self.cam_width)
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = int(self.cam_width * 4)
        depth_msg._data = array.array("B", depth.tobytes())

        info_msg = self._camera_info_template
        info_msg.header.stamp = stamp

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.cam_info_pub.publish(info_msg)

    def run_simulation(self) -> None:
        """Main real-time physics & rendering loop (runs until ROS shutdown or viewer close)."""
        dt = float(self.model.opt.timestep)
        js_period = 1.0 / max(1.0, self.joint_state_hz)
        cam_period = 1.0 / max(0.5, self.cam_fps)
        viewer_period = 1.0 / 30.0

        last_js = 0.0
        last_cam = 0.0
        last_viewer = 0.0

        viewer = None
        if not self.headless:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -28
            viewer.cam.distance = 1.85
            viewer.cam.lookat[:] = [0.35, 0.0, 0.55]

        try:
            wall_origin = time.perf_counter() - float(self.data.time)
            while rclpy.ok() and (viewer is None or viewer.is_running()):
                now = time.perf_counter()
                target_sim_time = now - wall_origin
                steps_run = 0
                with self._lock:
                    while self.data.time < target_sim_time and steps_run < 250:
                        self._step_trajectories_locked()
                        mujoco.mj_step(self.model, self.data)
                        steps_run += 1

                if now - last_js >= js_period:
                    self._publish_joint_states()
                    last_js = now

                if now - last_cam >= cam_period:
                    self._publish_camera_rgbd()
                    last_cam = now

                if viewer is not None and (now - last_viewer >= viewer_period):
                    viewer.sync()
                    last_viewer = now

                sleep_for = (float(self.data.time) + wall_origin) - time.perf_counter()
                if sleep_for > 0.0:
                    time.sleep(min(sleep_for, dt))
        finally:
            if viewer is not None:
                viewer.close()
            self.renderer.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MuJoCoSimNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_simulation()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
