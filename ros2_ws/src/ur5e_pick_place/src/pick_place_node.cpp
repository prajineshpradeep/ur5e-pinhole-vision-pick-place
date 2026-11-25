#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

using namespace std::chrono_literals;
using moveit::planning_interface::MoveGroupInterface;
using moveit::planning_interface::PlanningSceneInterface;

namespace
{
constexpr double kHoverZ = 0.61;
constexpr double kMinGraspZ = 0.426;
constexpr double kPlaceZ = 0.436;

geometry_msgs::msg::Pose makeDownwardPose(double x, double y, double z)
{
  geometry_msgs::msg::Pose p;
  p.position.x = x;
  p.position.y = y;
  p.position.z = z;
  p.orientation.x = 1.0;
  p.orientation.y = 0.0;
  p.orientation.z = 0.0;
  p.orientation.w = 0.0;
  return p;
}

void addStaticEnvironment(PlanningSceneInterface& scene)
{
  std::vector<moveit_msgs::msg::CollisionObject> objects;

  // Worktable: top surface at z = 0.397 (3mm tolerance below physical 0.400 for gravity sag)
  moveit_msgs::msg::CollisionObject table;
  table.header.frame_id = "world";
  table.id = "worktable";
  shape_msgs::msg::SolidPrimitive table_box;
  table_box.type = shape_msgs::msg::SolidPrimitive::BOX;
  table_box.dimensions = {0.72, 0.88, 0.04};
  geometry_msgs::msg::Pose table_pose;
  table_pose.position.x = 0.52;
  table_pose.position.y = 0.0;
  table_pose.position.z = 0.377;
  table_pose.orientation.w = 1.0;
  table.primitives.push_back(table_box);
  table.primitive_poses.push_back(table_pose);
  table.operation = moveit_msgs::msg::CollisionObject::ADD;
  objects.push_back(table);

  // Target tray base slab
  moveit_msgs::msg::CollisionObject tray;
  tray.header.frame_id = "world";
  tray.id = "target_tray_base";
  shape_msgs::msg::SolidPrimitive tray_box;
  tray_box.type = shape_msgs::msg::SolidPrimitive::BOX;
  tray_box.dimensions = {0.26, 0.26, 0.008};
  geometry_msgs::msg::Pose tray_pose;
  tray_pose.position.x = 0.48;
  tray_pose.position.y = 0.23;
  tray_pose.position.z = 0.401;
  tray_pose.orientation.w = 1.0;
  tray.primitives.push_back(tray_box);
  tray.primitive_poses.push_back(tray_pose);
  tray.operation = moveit_msgs::msg::CollisionObject::ADD;
  objects.push_back(tray);

  scene.applyCollisionObjects(objects);
}

bool moveToJoints(MoveGroupInterface& group, const std::vector<double>& joints, const rclcpp::Logger& log)
{
  group.setStartStateToCurrentState();
  group.setJointValueTarget(joints);
  MoveGroupInterface::Plan plan;
  if (group.plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(log, "Joint planning failed");
    return false;
  }
  if (group.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(log, "Joint execution failed");
    return false;
  }
  return true;
}

bool moveToPose(MoveGroupInterface& group, const geometry_msgs::msg::Pose& target, const rclcpp::Logger& log)
{
  group.setStartStateToCurrentState();
  group.setPoseTarget(target);
  MoveGroupInterface::Plan plan;
  if (group.plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(log, "Pose planning failed for (%.3f, %.3f, %.3f)",
                 target.position.x, target.position.y, target.position.z);
    return false;
  }
  if (group.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(log, "Pose execution failed");
    return false;
  }
  return true;
}

bool moveCartesian(MoveGroupInterface& group, const geometry_msgs::msg::Pose& target, const rclcpp::Logger& log)
{
  group.setStartStateToCurrentState();
  std::vector<geometry_msgs::msg::Pose> waypoints{target};
  moveit_msgs::msg::RobotTrajectory traj;
  const double fraction = group.computeCartesianPath(waypoints, 0.005, 0.0, traj, true);
  if (fraction < 0.90) {
    RCLCPP_WARN(log, "Cartesian path fraction low (%.2f) to (%.3f, %.3f, %.3f); falling back to OMPL",
                fraction, target.position.x, target.position.y, target.position.z);
    return moveToPose(group, target, log);
  }
  if (group.execute(traj) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(log, "Cartesian execution failed");
    return false;
  }
  return true;
}

bool setGripper(MoveGroupInterface& gripper, const std::string& named_state, const rclcpp::Logger& log)
{
  gripper.setStartStateToCurrentState();
  gripper.setNamedTarget(named_state);
  MoveGroupInterface::Plan plan;
  if (gripper.plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(log, "Gripper plan failed for '%s'", named_state.c_str());
    return false;
  }
  if (gripper.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(log, "Gripper execution failed for '%s'", named_state.c_str());
    return false;
  }
  return true;
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opts;
  opts.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("pick_place_node", opts);
  const auto log = node->get_logger();

  std::mutex pose_mutex;
  geometry_msgs::msg::PoseArray latest_poses;
  rclcpp::Time latest_pose_rx_time(0, 0, RCL_ROS_TIME);
  auto pose_sub = node->create_subscription<geometry_msgs::msg::PoseArray>(
      "/detected_objects/poses", 10,
      [&](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(pose_mutex);
        latest_poses = *msg;
        latest_pose_rx_time = node->now();
      });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  MoveGroupInterface arm(node, "arm");
  MoveGroupInterface gripper(node, "gripper");
  PlanningSceneInterface scene;

  arm.setPoseReferenceFrame("world");
  arm.setEndEffectorLink("pinch_tcp");
  arm.setMaxVelocityScalingFactor(0.55);
  arm.setMaxAccelerationScalingFactor(0.55);
  arm.setPlanningTime(5.0);
  arm.setNumPlanningAttempts(10);

  gripper.setMaxVelocityScalingFactor(0.8);
  gripper.setMaxAccelerationScalingFactor(0.8);

  // Wait briefly for current robot state from /joint_states
  RCLCPP_INFO(log, "Waiting for MoveIt current robot state...");
  while (rclcpp::ok() && !arm.getCurrentState(2.0)) {
    RCLCPP_INFO(log, "Still waiting for /joint_states...");
  }

  addStaticEnvironment(scene);
  rclcpp::sleep_for(500ms);

  // Eye-in-hand scan configuration looking cleanly over the pick zone
  const std::vector<double> scan_joints = {-0.60, -1.65, 0.70, -0.6208, -1.5708, 0.0};
  const std::vector<geometry_msgs::msg::Pose> place_slots = {
      makeDownwardPose(0.425, 0.175, kPlaceZ),
      makeDownwardPose(0.535, 0.175, kPlaceZ),
      makeDownwardPose(0.425, 0.285, kPlaceZ),
      makeDownwardPose(0.535, 0.285, kPlaceZ),
  };

  auto waitForFreshDetections = [&](std::vector<geometry_msgs::msg::Pose>& out_poses) -> bool {
    const auto trigger_time = node->now();
    const auto deadline = trigger_time + rclcpp::Duration::from_seconds(4.0);
    while (rclcpp::ok() && node->now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(pose_mutex);
        if (latest_pose_rx_time > trigger_time && !latest_poses.poses.empty()) {
          out_poses = latest_poses.poses;
          return true;
        }
      }
      rclcpp::sleep_for(100ms);
    }
    std::lock_guard<std::mutex> lock(pose_mutex);
    out_poses = latest_poses.poses;
    return !out_poses.empty();
  };

  size_t placed_count = 0;
  while (rclcpp::ok() && placed_count < place_slots.size()) {
    RCLCPP_INFO(log, "Moving to eye-in-hand scan pose (cycle %zu)...", placed_count);
    if (!setGripper(gripper, "open", log) || !moveToJoints(arm, scan_joints, log)) {
      break;
    }

    rclcpp::sleep_for(400ms);
    std::vector<geometry_msgs::msg::Pose> detected;
    if (!waitForFreshDetections(detected)) {
      RCLCPP_INFO(log, "No more objects detected in pick zone. Pick-and-place finished!");
      break;
    }

    const auto target = detected.front();
    RCLCPP_INFO(log, "Selected target #%zu at (%.4f, %.4f, %.4f)",
                placed_count, target.position.x, target.position.y, target.position.z);

    auto pre_pick = makeDownwardPose(target.position.x, target.position.y, std::max(0.56, target.position.z + 0.14));
    auto pick_pose = makeDownwardPose(target.position.x, target.position.y, std::max(kMinGraspZ, target.position.z));
    auto lift_pose = makeDownwardPose(target.position.x, target.position.y, kHoverZ);

    const auto& slot = place_slots[placed_count];
    auto above_slot = makeDownwardPose(slot.position.x, slot.position.y, kHoverZ);

    if (!moveCartesian(arm, pre_pick, log)) {
      break;
    }
    if (!moveCartesian(arm, pick_pose, log)) {
      break;
    }

    // Register held object in MoveIt PlanningScene and attach before closing fingers
    moveit_msgs::msg::CollisionObject held_obj;
    held_obj.header.frame_id = "world";
    held_obj.id = "held_obj";
    shape_msgs::msg::SolidPrimitive obj_prim;
    obj_prim.type = shape_msgs::msg::SolidPrimitive::BOX;
    obj_prim.dimensions = {0.042, 0.042, 0.042};
    held_obj.primitives.push_back(obj_prim);
    held_obj.primitive_poses.push_back(pick_pose);
    held_obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    scene.applyCollisionObjects({held_obj});
    arm.attachObject("held_obj", "pinch_tcp", {"left_finger", "right_finger", "gripper_base"});

    if (!setGripper(gripper, "close", log)) {
      break;
    }
    rclcpp::sleep_for(250ms);

    if (!moveCartesian(arm, lift_pose, log)) {
      break;
    }
    if (!moveCartesian(arm, above_slot, log)) {
      break;
    }
    if (!moveCartesian(arm, slot, log)) {
      break;
    }

    if (!setGripper(gripper, "open", log)) {
      break;
    }
    rclcpp::sleep_for(200ms);
    arm.detachObject("held_obj");
    scene.removeCollisionObjects({"held_obj"});

    if (!moveCartesian(arm, above_slot, log)) {
      break;
    }

    ++placed_count;
    RCLCPP_INFO(log, "Successfully placed object #%zu into tray slot!", placed_count);
  }

  RCLCPP_INFO(log, "Returning arm to home pose...");
  arm.setStartStateToCurrentState();
  arm.setNamedTarget("home");
  arm.move();
  RCLCPP_INFO(log, "Pick-and-place complete! Total objects placed: %zu", placed_count);

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
