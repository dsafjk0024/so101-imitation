#!/usr/bin/env bash
# Source this (don't execute) before any `ros2 run`/`ros2 launch` in this repo:
#     source scripts/ros_env.sh
#
# Sets up ROS jazzy + this workspace's install space, and strips any conda env
# lib dirs from LD_LIBRARY_PATH. Those shadow ROS's own shared libraries --
# confirmed to crash robot_state_publisher (missing liburdfdom_model.so.4.0)
# and usb_cam_node_exe (libgcc_s/libhwy version mismatch) when present. Set
# CONDA_ENV_LIBS to a colon-separated list if your shell adds different ones.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/install/setup.bash"

CONDA_ENV_LIBS="${CONDA_ENV_LIBS:-$HOME/miniconda3/envs}"
CLEANED="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -F -f <(echo "$CONDA_ENV_LIBS" | tr ':' '\n') | paste -sd: -)"
export LD_LIBRARY_PATH="$CLEANED"

# Fast-DDS's default shared-memory transport uses a ~512KB segment in
# /dev/shm; a single 640x480 rgb8 camera frame (~900KB) doesn't fit, causing
# multi-second stalls and rate collapse on image_raw topics (confirmed by
# reproducing the stall with a bare large-message publisher, no camera
# involved, and clearing it by forcing UDP-only). Loopback UDP has no such
# per-sample size limit.
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
