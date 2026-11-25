# UR5e Eye-in-Hand Pinhole Vision Pick-and-Place (`ROS 2 Humble` + `MoveIt 2` + `MuJoCo`)

Autonomous tabletop pick-and-place pipeline combining the official UR5e URDF, a custom parallel-jaw gripper with a wrist-mounted pinhole RGB-D camera (`wrist_cam`), MoveIt 2 motion planning in C++17, TF2-based 3D pinhole object localization, and MuJoCo 3.15 physics simulation.

For a complete technical breakdown, diagrams, and verified test results, see [UR5e_Pick_and_Place_Project_Report.pdf](UR5e_Pick_and_Place_Project_Report.pdf).

## Repository Structure

| Path | Description |
| --- | --- |
| `ros2_ws/src/Universal_Robots_ROS2_Description` | Upstream UR5e URDF/Xacro & meshes (`humble` submodule) |
| `ros2_ws/src/ur5e_custom_gripper_moveit_config` | UR5e + pedestal + custom parallel-jaw gripper URDF/Xacro (`pinch_tcp`, `wrist_cam_optical_frame`) |
| `ros2_ws/src/ur5e_moveit_config` | MoveIt 2 SRDF (`arm` / `gripper` groups), kinematics, joint limits, and controller configs |
| `ros2_ws/src/ur5e_mujoco_sim` | MuJoCo scene (`ur5e_scene.xml`), ROS 2 physics/camera bridge (`mujoco_node`), and HSV + pinhole unprojection detector (`vision_node`) |
| `ros2_ws/src/ur5e_pick_place` | C++ `MoveGroupInterface` + `PlanningSceneInterface` pick-and-place node (`pick_place_node`) and unified launch file |
| `UR5e_Pick_and_Place_Project_Report.pdf` | Full technical project report |

## Prerequisites

- Ubuntu 22.04 + **ROS 2 Humble** + **MoveIt 2** (`ros-humble-moveit`, `ros-humble-xacro`, `ros-humble-tf2-ros`)
- Python 3.10 virtual environment at `.venv/` with `mujoco` (`>= 3.15`), `numpy`, and `glfw`

## Build & Run

```bash
# Clone with upstream URDF submodule
git clone --recurse-submodules https://github.com/prajineshpradeep/ur5e-pinhole-vision-pick-place.git
cd ur5e-pinhole-vision-pick-place/ros2_ws

# Build ROS 2 packages
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Launch MuJoCo + RViz2 + MoveIt 2 + Vision + C++ Pick-and-Place
source install/setup.bash
ros2 launch ur5e_pick_place pick_and_place.launch.py
```

Headless test mode (no GUI / RViz):

```bash
ros2 launch ur5e_pick_place pick_and_place.launch.py headless:=true use_rviz:=false
```
