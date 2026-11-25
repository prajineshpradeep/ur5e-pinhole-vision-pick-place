"""Eye-in-hand pinhole camera object detector node for UR5e pick-and-place."""

from __future__ import annotations

import array
import json
import os
import sys

# Allow system ROS 2 python3 to import numpy from .venv if needed, while staying compatible with system numpy
_VENV_SITE = "/home/praji/mujoco_project/.venv/lib/python3.10/site-packages"
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.append(_VENV_SITE)

from geometry_msgs.msg import Pose, PoseArray
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_ros


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _rgb_to_hsv(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (H, W, 3) uint8 RGB image to (H_deg, S, V) in pure NumPy."""
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    nz = delta > 1e-6
    rmask = nz & (cmax == r)
    gmask = nz & (cmax == g)
    bmask = nz & (cmax == b)

    hue[rmask] = 60.0 * (((g[rmask] - b[rmask]) / delta[rmask]) % 6.0)
    hue[gmask] = 60.0 * (((b[gmask] - r[gmask]) / delta[gmask]) + 2.0)
    hue[bmask] = 60.0 * (((r[bmask] - g[bmask]) / delta[bmask]) + 4.0)
    sat = np.where(cmax > 1e-6, delta / (cmax + 1e-6), 0.0)
    return hue, sat, cmax


def _draw_box_and_cross(
    img: np.ndarray,
    u0: int,
    v0: int,
    u1: int,
    v1: int,
    uc: int,
    vc: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a 2px bounding box and center crosshair on an RGB uint8 image."""
    h, w, _ = img.shape
    u0c, u1c = max(0, min(w - 1, u0)), max(0, min(w - 1, u1))
    v0c, v1c = max(0, min(h - 1, v0)), max(0, min(h - 1, v1))
    img[v0c : min(h, v0c + 2), u0c : u1c + 1] = color
    img[max(0, v1c - 1) : v1c + 1, u0c : u1c + 1] = color
    img[v0c : v1c + 1, u0c : min(w, u0c + 2)] = color
    img[v0c : v1c + 1, max(0, u1c - 1) : u1c + 1] = color

    if 0 <= vc < h:
        img[vc, max(0, uc - 6) : min(w, uc + 7)] = (255, 255, 255)
    if 0 <= uc < w:
        img[max(0, vc - 6) : min(h, vc + 7), uc] = (255, 255, 255)


class PinholeVisionNode(Node):
    """Detects tabletop pick objects from /wrist_cam/* using pinhole unprojection + TF2."""

    _COLOR_SPECS = [
        ("red_cube", lambda h: (h < 16.0) | (h > 340.0), 0.48, 0.25, (255, 80, 80)),
        ("green_cube", lambda h: (h >= 85.0) & (h <= 155.0), 0.42, 0.25, (80, 255, 110)),
        ("blue_cylinder", lambda h: (h >= 180.0) & (h <= 255.0), 0.35, 0.25, (90, 170, 255)),
        ("yellow_cube", lambda h: (h >= 38.0) & (h <= 68.0), 0.48, 0.35, (255, 225, 70)),
    ]

    def __init__(self) -> None:
        super().__init__("pinhole_vision_node")

        self.declare_parameter("camera_frame", "wrist_cam_optical_frame")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("min_area_px", 60)

        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.world_frame = self.get_parameter("world_frame").get_parameter_value().string_value
        self.min_area_px = self.get_parameter("min_area_px").get_parameter_value().integer_value

        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None

        self._latest_rgb: tuple[tuple[int, int], np.ndarray] | None = None
        self._latest_depth: tuple[tuple[int, int], np.ndarray] | None = None
        self._last_processed_stamp: tuple[int, int] | None = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, "/wrist_cam/camera_info", self._on_camera_info, 10)
        self.create_subscription(Image, "/wrist_cam/image_raw", self._on_rgb, 10)
        self.create_subscription(Image, "/wrist_cam/depth/image_raw", self._on_depth, 10)

        self.poses_pub = self.create_publisher(PoseArray, "/detected_objects/poses", 10)
        self.labels_pub = self.create_publisher(String, "/detected_objects/labels", 10)
        self.annot_pub = self.create_publisher(Image, "/wrist_cam/image_annotated", 10)

        self.get_logger().info("Eye-in-hand PinholeVisionNode listening on /wrist_cam/*")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])

    def _on_rgb(self, msg: Image) -> None:
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        stamp_key = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        self._latest_rgb = (stamp_key, rgb)
        self._try_process()

    def _on_depth(self, msg: Image) -> None:
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
        stamp_key = (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        self._latest_depth = (stamp_key, depth)
        self._try_process()

    def _unproject(self, u: np.ndarray | float, v: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        assert self._fx is not None and self._fy is not None and self._cx is not None and self._cy is not None
        x_c = (u - self._cx) * z / self._fx
        y_c = (v - self._cy) * z / self._fy
        if np.ndim(x_c) == 0:
            return np.array([float(x_c), float(y_c), float(z)], dtype=np.float64)
        return np.column_stack([x_c, y_c, z]).astype(np.float64)

    def _try_process(self) -> None:
        if self._fx is None or self._latest_rgb is None or self._latest_depth is None:
            return
        rgb_stamp, rgb = self._latest_rgb
        depth_stamp, depth = self._latest_depth
        if rgb_stamp == self._last_processed_stamp:
            return
        # Require closely matched RGB & depth frames (exact stamp in mujoco_node)
        dt_ns = abs(
            (rgb_stamp[0] - depth_stamp[0]) * 1_000_000_000 + (rgb_stamp[1] - depth_stamp[1])
        )
        if dt_ns > 50_000_000:
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                rclpy.time.Time(seconds=rgb_stamp[0], nanoseconds=rgb_stamp[1]),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.world_frame, self.camera_frame, rclpy.time.Time()
                )
            except Exception:
                return

        self._last_processed_stamp = rgb_stamp
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        t_wc = np.array([t.x, t.y, t.z], dtype=np.float64)
        r_wc = _quat_to_rot(q.x, q.y, q.z, q.w)

        hue, sat, val = _rgb_to_hsv(rgb)
        h_img, w_img = depth.shape
        any_color_mask = np.zeros((h_img, w_img), dtype=bool)
        color_masks: dict[str, tuple[np.ndarray, tuple[int, int, int]]] = {}
        for label, hue_fn, min_s, min_v, draw_rgb in self._COLOR_SPECS:
            m = hue_fn(hue) & (sat >= min_s) & (val >= min_v) & (depth > 0.12) & (depth < 3.0)
            color_masks[label] = (m, draw_rgb)
            any_color_mask |= m

        pose_array = PoseArray()
        pose_array.header.stamp.sec = rgb_stamp[0]
        pose_array.header.stamp.nanosec = rgb_stamp[1]
        pose_array.header.frame_id = self.world_frame

        detections_meta: list[dict] = []
        annotated = rgb.copy()

        for label, (mask, draw_rgb) in color_masks.items():
            if int(mask.sum()) < self.min_area_px:
                continue

            vs, us = np.where(mask)
            depths_cand = depth[vs, us].astype(np.float64)
            pts_cam = self._unproject(us.astype(np.float64), vs.astype(np.float64), depths_cand)
            pts_world = pts_cam @ r_wc.T + t_wc

            # Keep only objects resting on the pick zone of the worktable (rejects robot links)
            in_pick_zone = (
                (pts_world[:, 0] >= 0.24)
                & (pts_world[:, 0] <= 0.80)
                & (pts_world[:, 1] >= -0.38)
                & (pts_world[:, 1] <= 0.06)
                & (pts_world[:, 2] >= 0.39)
                & (pts_world[:, 2] <= 0.51)
            )
            us, vs = us[in_pick_zone], vs[in_pick_zone]
            if len(us) < self.min_area_px:
                continue

            u_med, v_med = float(np.median(us)), float(np.median(vs))
            inlier = (np.abs(us - u_med) < 60) & (np.abs(vs - v_med) < 60)
            us, vs = us[inlier], vs[inlier]
            if len(us) < self.min_area_px:
                continue

            u_c, v_c = float(np.mean(us)), float(np.mean(vs))
            u_min, u_max = int(us.min()), int(us.max())
            v_min, v_max = int(vs.min()), int(vs.max())

            dist2 = (us - u_c) ** 2 + (vs - v_c) ** 2
            core_n = max(10, len(us) // 3)
            core_idx = np.argpartition(dist2, min(core_n, len(us) - 1))[:core_n]
            depth_top = float(np.percentile(depth[vs[core_idx], us[core_idx]], 25))

            p_top_world = r_wc @ self._unproject(u_c, v_c, depth_top) + t_wc

            # Estimate surrounding table Z from nearby non-object pixels
            pad = 14
            ru0, ru1 = max(0, u_min - pad), min(w_img, u_max + pad + 1)
            rv0, rv1 = max(0, v_min - pad), min(h_img, v_max + pad + 1)
            bg_mask = ~any_color_mask[rv0:rv1, ru0:ru1]
            bg_depths = depth[rv0:rv1, ru0:ru1][bg_mask]
            valid_bg = bg_depths[(bg_depths > depth_top + 0.015) & (bg_depths < depth_top + 0.20)]
            if len(valid_bg) >= 15:
                depth_table = float(np.median(valid_bg))
                p_table_world = r_wc @ self._unproject(u_c, v_c, depth_table) + t_wc
                table_z = float(p_table_world[2])
                est_height = float(np.clip(p_top_world[2] - table_z, 0.025, 0.09))
            else:
                est_height = 0.042
                table_z = float(p_top_world[2]) - est_height

            p_grasp = p_top_world.copy()
            p_grasp[2] = table_z + 0.50 * est_height

            pose = Pose()
            pose.position.x = float(p_grasp[0])
            pose.position.y = float(p_grasp[1])
            pose.position.z = float(p_grasp[2])
            # Downward gripper orientation (180 deg around world X)
            pose.orientation.x = 1.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0
            pose_array.poses.append(pose)

            detections_meta.append(
                {
                    "label": label,
                    "x": round(float(p_grasp[0]), 4),
                    "y": round(float(p_grasp[1]), 4),
                    "z": round(float(p_grasp[2]), 4),
                    "top_z": round(float(p_top_world[2]), 4),
                    "height_m": round(est_height, 4),
                    "depth_m": round(depth_top, 4),
                    "pixel_center": [round(u_c, 1), round(v_c, 1)],
                }
            )
            _draw_box_and_cross(
                annotated, u_min - 2, v_min - 2, u_max + 2, v_max + 2, int(round(u_c)), int(round(v_c)), draw_rgb
            )

        self.poses_pub.publish(pose_array)
        labels_msg = String()
        labels_msg.data = json.dumps(detections_meta)
        self.labels_pub.publish(labels_msg)

        annot_msg = Image()
        annot_msg.header = pose_array.header
        annot_msg.header.frame_id = self.camera_frame
        annot_msg.height = int(h_img)
        annot_msg.width = int(w_img)
        annot_msg.encoding = "rgb8"
        annot_msg.is_bigendian = 0
        annot_msg.step = int(w_img * 3)
        annot_msg._data = array.array("B", annotated.tobytes())
        self.annot_pub.publish(annot_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PinholeVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
