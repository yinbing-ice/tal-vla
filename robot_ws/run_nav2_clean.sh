#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
export PATH=/usr/bin:/bin:/usr/sbin:/sbin

set +u
# 2026-05-30 修改：当前机器实际安装的是 ROS2 Jazzy，不是 Humble。
source /opt/ros/jazzy/setup.bash
source /root/gpufree-data/code/tal-vla/robot_ws/install/local_setup.bash
set -u

# 2026-05-30 修改：当前工程的最小 Nav2 工作区已经迁到 code/tal-vla/robot_ws，
# 这里统一切到新路径，避免继续 source 旧工程里的 install 和地图配置。

# 2026-05-30 修改：这里直接使用当前环境里的 ros2，避免继续绑定到旧的 Humble 绝对路径。
exec ros2 launch robot_navigation nav2_launch.py \
  mode:=isaac \
  use_sim_time:=true \
  map:=/root/gpufree-data/code/tal-vla/robot_ws/src/robot_navigation/maps/expff_map.yaml \
  params_file:=/root/gpufree-data/code/tal-vla/robot_ws/src/robot_navigation/config/nav2_params.yaml \
  "$@"
