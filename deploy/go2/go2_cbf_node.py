#!/usr/bin/env python3
"""ROS2 node: NMR-CBF / PSHR-CBF safety filter for Unitree Go2.

Subscribes : nav_msgs/Odometry   (world-frame pose + velocity)
Publishes  : unitree_go/msg/SportModeCmd  (body-frame velocity commands)

The CBF filter computes safe 2D accelerations [ax, ay] in the world frame,
integrates them to velocity commands, then rotates to the body frame using
the robot's current heading from odometry.

Usage
-----
    ros2 run <your_package> go2_cbf_node.py \
        --ros-args \
        -p filter_type:=nmr \
        -p phi_weights_path:=/path/to/phi_params.pkl \
        -p goal:=[2.0,0.0,0.0,0.0] \
        -p obs_radius:=0.25

Parameters
----------
filter_type       : str   'nmr' (default) or 'pshr'
phi_weights_path  : str   path to phi_params.pkl  (required if filter_type=nmr)
state_eps         : list  per-dim uncertainty [px,vx,py,vy]  default [0.3,0.15,0.3,0.15]
goal              : list  goal state [px,vx,py,vy]           default [2.,0.,0.,0.]
obs_radius        : float obstacle radius (m)                default 0.25
control_hz        : float control loop rate                  default 20.0
odom_topic        : str   odometry input topic               default /odom
cmd_topic         : str   SportModeCmd output topic          default /sport_mode_cmd
"""

import math
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

try:
    from unitree_go.msg import SportModeCmd
except ImportError:
    # Provide a stub so the file can be imported for unit-testing without the
    # unitree_go package installed.
    class SportModeCmd:  # type: ignore[no-redef]
        def __init__(self):
            self.mode = 0
            self.gait_type = 0
            self.velocity = [0.0, 0.0, 0.0]
            self.yaw_speed = 0.0

from go2_cbf_filter import (
    DT,
    DEFAULT_STATE_EPS,
    ALPHA_QP,
    NmrCbfFilter,
    PshrCbfFilter,
    nom_pol_goto,
)


def _quat_to_yaw(qx, qy, qz, qw) -> float:
    """Extract yaw (rotation about world Z) from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _world_to_body(vx_w: float, vy_w: float, yaw: float):
    """Rotate a 2D world-frame vector to the robot body frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    vx_b =  c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b


class Go2CbfNode(Node):

    def __init__(self):
        super().__init__("go2_cbf_node")

        # ── Declare parameters ────────────────────────────────────────────────
        self.declare_parameter("filter_type",      "nmr")
        self.declare_parameter("phi_weights_path", "")
        self.declare_parameter("state_eps",        [0.3, 0.15, 0.3, 0.15])
        self.declare_parameter("goal",             [2.0, 0.0, 0.0, 0.0])
        self.declare_parameter("obs_radius",       0.25)
        self.declare_parameter("control_hz",       20.0)
        self.declare_parameter("odom_topic",       "/odom")
        self.declare_parameter("cmd_topic",        "/sport_mode_cmd")

        # ── Read parameters ───────────────────────────────────────────────────
        filter_type      = self.get_parameter("filter_type").value
        phi_weights_path = self.get_parameter("phi_weights_path").value
        state_eps        = list(self.get_parameter("state_eps").value)
        self._goal       = jnp.array(self.get_parameter("goal").value,
                                     dtype=jnp.float32)
        control_hz       = float(self.get_parameter("control_hz").value)
        odom_topic       = self.get_parameter("odom_topic").value
        cmd_topic        = self.get_parameter("cmd_topic").value

        # Patch the module-level R_OBS if obs_radius differs from the default.
        import go2_cbf_filter as _f
        _f.R_OBS = float(self.get_parameter("obs_radius").value)

        # ── Build filter ──────────────────────────────────────────────────────
        self._filter_type = filter_type.lower()
        if self._filter_type == "nmr":
            if not phi_weights_path:
                self.get_logger().fatal(
                    "filter_type=nmr requires phi_weights_path to be set."
                )
                sys.exit(1)
            self.get_logger().info(f"Loading NMR-CBF weights from: {phi_weights_path}")
            self._filter = NmrCbfFilter(
                phi_weights_path,
                state_eps=jnp.array(state_eps, dtype=jnp.float32),
                alpha=ALPHA_QP,
            )
        elif self._filter_type == "pshr":
            self.get_logger().info("Initialising PSHR-CBF (oracle, no network).")
            self._filter = PshrCbfFilter(
                state_eps=jnp.array(state_eps, dtype=jnp.float32),
                alpha=ALPHA_QP,
            )
            self._pshr_key = jr.PRNGKey(0)
        else:
            self.get_logger().fatal(
                f"Unknown filter_type '{filter_type}'. Use 'nmr' or 'pshr'."
            )
            sys.exit(1)

        # ── Latest odometry (updated by subscriber callback) ──────────────────
        self._px  = 0.0
        self._vx  = 0.0
        self._py  = 0.0
        self._vy  = 0.0
        self._yaw = 0.0
        self._odom_received = False

        # ── ROS2 pub/sub/timer ────────────────────────────────────────────────
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_callback, 10
        )
        self._cmd_pub = self.create_publisher(SportModeCmd, cmd_topic, 10)

        # Warm up JIT before the control loop starts.
        self.get_logger().info("Compiling JAX JIT (may take ~30 s on first run)…")
        t0 = time.time()
        self._filter.warmup()
        self.get_logger().info(f"JIT done in {time.time()-t0:.1f} s.")

        period = 1.0 / control_hz
        self._timer = self.create_timer(period, self._control_callback)
        self.get_logger().info(
            f"Go2 CBF node ready ({filter_type.upper()}, {control_hz:.0f} Hz)."
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        ori = msg.pose.pose.orientation

        self._px  = pos.x
        self._py  = pos.y
        self._vx  = vel.x
        self._vy  = vel.y
        self._yaw = _quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
        self._odom_received = True

    def _control_callback(self):
        if not self._odom_received:
            return

        xhat  = jnp.array([self._px, self._vx, self._py, self._vy],
                           dtype=jnp.float32)
        u_nom = nom_pol_goto(xhat, self._goal)

        # ── Run CBF filter ─────────────────────────────────────────────────────
        if self._filter_type == "nmr":
            u_cbf = self._filter.compute_control(xhat, u_nom)
        else:
            self._pshr_key, subkey = jr.split(self._pshr_key)
            u_cbf = self._filter.compute_control(xhat, u_nom, subkey)

        ax, ay = float(u_cbf[0]), float(u_cbf[1])

        # ── Integrate acceleration → velocity (world frame) ────────────────────
        vx_cmd = self._vx + ax * DT
        vy_cmd = self._vy + ay * DT

        # ── Rotate to body frame ───────────────────────────────────────────────
        vx_body, vy_body = _world_to_body(vx_cmd, vy_cmd, self._yaw)

        # ── Publish ────────────────────────────────────────────────────────────
        cmd = SportModeCmd()
        cmd.mode      = 2   # walking/running mode
        cmd.gait_type = 1   # trot
        cmd.velocity  = [vx_body, vy_body, 0.0]
        cmd.yaw_speed = 0.0
        self._cmd_pub.publish(cmd)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = Go2CbfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
