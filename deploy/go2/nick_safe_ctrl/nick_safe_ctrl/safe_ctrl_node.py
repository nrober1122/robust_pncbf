"""Safety-filtered Unitree Go2 controller -- pure CBF obstacle avoidance.

Per control tick:
  1. SportModeState -> robot pose (x, y, yaw), converted into a task frame
     captured from the robot's pose at startup.
  2. A control point is held ``d`` ahead of the robot (feedback linearization).
  3. The 2-D double-integrator CBF runs on the control point: state
     ``(xc, vxc, yc, vyc)`` with ``(vxc, vyc)`` the internally integrated
     control-point velocity.
  4. A goal-seeking PD nominal acceleration is filtered by the min-norm CBF-QP.
  5. The safe acceleration is integrated -> ``(vxc, vyc)`` and mapped through
     the linearizing transform -> ``(vx, vyaw)``.
  6. A Unitree sport-mode ``Move`` request is published on /api/sport/request.

The learned NMR-CBF network is intentionally NOT used here -- this is the
pure-CBF baseline (phi = 0).  ``value_function.py`` is left in place; re-enabling
the network is a one-line change in ``double_integrator_cbf.cbf_qp_filter``
(add ``phi / alpha_qp`` to ``psi1``).  Because the NN path is removed, this node
does not import jax/flax and runs with only numpy + rclpy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from unitree_api.msg import Request
from unitree_go.msg import SportModeState

from . import diagnostics
from . import double_integrator_cbf as dic
from . import phi_sources
from .feedback_linearization import control_point, to_unicycle_cmd

# Unitree sport-mode API ids (see ros2_sport_client.h).
SPORT_API_ID_MOVE = 1008
SPORT_API_ID_STOPMOVE = 1003


class SafeCtrlNode(Node):
    def __init__(self) -> None:
        super().__init__("safe_ctrl_node")

        # ---- Parameters ------------------------------------------------------
        self.declare_parameter("state_topic", "lf/sportmodestate")
        self.declare_parameter("sport_request_topic", "/api/sport/request")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("state_timeout_sec", 0.5)
        self.declare_parameter("dry_run", True)

        # Goal + obstacle, in the startup task frame (metres). x is "ahead",
        # y is "left" of where the robot stood when the node started.
        self.declare_parameter("goal_xy", [3.6, 0.0])
        self.declare_parameter("goal_tolerance", 0.15)
        # Obstacles are parallel arrays (ROS2 params can't nest lists): the
        # i-th obstacle is centred at (obstacles_x[i], obstacles_y[i]) with real
        # radius obstacle_radii[i].  body_margin is shared.  Each control tick
        # only the nearest n_closest_obs obstacles become CBF-QP constraints.
        self.declare_parameter("obstacles_x", [2.0])
        self.declare_parameter("obstacles_y", [0.35])
        self.declare_parameter("obstacle_radii", [0.30])
        self.declare_parameter("body_margin", 0.20)
        self.declare_parameter("n_closest_obs", 2)

        # Feedback linearization + CBF.
        self.declare_parameter("control_point_distance", 0.5)
        self.declare_parameter("alpha0", 2.0)
        self.declare_parameter("alpha_qp", 1.0)
        self.declare_parameter("accel_max", 0.5)
        self.declare_parameter("qp_penalty", 10.0)
        self.declare_parameter("nominal_kp", 2.0)
        self.declare_parameter("nominal_kd", 2.0)

        # CBF method (phi-source) selection and synthetic measurement noise.
        self.declare_parameter("cbf_method", "plain")
        self.declare_parameter("state_eps", [0.10, 0.05, 0.10, 0.05])
        self.declare_parameter("noise_injection.enabled", False)
        self.declare_parameter("noise_injection.bias", [0.0, 0.0, 0.0, 0.0])
        # PSHR-CBF projected-gradient ascent (notebook defaults).
        self.declare_parameter("pshr_restarts", 4)
        self.declare_parameter("pshr_steps", 25)
        self.declare_parameter("pshr_lr", 0.05)
        self.declare_parameter("pshr_seed", 0)
        # NMR-CBF: pickled Flax PhiNet weights (loaded into the numpy PhiNet).
        self.declare_parameter("nmrcbf_checkpoint", "ckpts/phi_params.pkl")
        # MR-CBF Lipschitz estimation domain (only used when cbf_method=mrcbf).
        self.declare_parameter("mrcbf_lipschitz_lo", [-3.0, -2.0, -3.0, -2.0])
        self.declare_parameter("mrcbf_lipschitz_hi", [ 3.0,  2.0,  3.0,  2.0])
        self.declare_parameter("mrcbf_lipschitz_samples", 5000)

        # Command limits (conservative -- raise once verified).
        self.declare_parameter("cp_speed_max", 0.4)
        self.declare_parameter("vx_min", -0.15)
        self.declare_parameter("vx_max", 0.4)
        self.declare_parameter("vyaw_max", 0.8)

        gp = self.get_parameter
        self._dry_run = bool(gp("dry_run").value)
        self._rate = float(gp("control_rate_hz").value)
        self._state_timeout = float(gp("state_timeout_sec").value)
        self._d = float(gp("control_point_distance").value)
        if self._d <= 0.0:
            raise ValueError("control_point_distance must be > 0")
        self._goal = np.array(gp("goal_xy").value, dtype=float)
        self._goal_tol = float(gp("goal_tolerance").value)
        self._accel_max = float(gp("accel_max").value)
        self._kp = float(gp("nominal_kp").value)
        self._kd = float(gp("nominal_kd").value)
        self._cp_speed_max = float(gp("cp_speed_max").value)
        self._vx_min = float(gp("vx_min").value)
        self._vx_max = float(gp("vx_max").value)
        self._vyaw_max = float(gp("vyaw_max").value)

        # The CBF protects the control point, which rides d ahead of the body.
        # We deliberately do NOT inflate the keep-out radius by d: the control
        # point leads the body, so guarding it (rather than the body) leaves up
        # to d of slack on the body in the worst case. That worst case only
        # arises when the robot drives straight at the obstacle; in practice the
        # body swerves with the control point and d-of-penetration is not a
        # practical concern, so R_cbf = obstacle_radius + body_margin only.
        #
        # Obstacles come as parallel arrays; gains are shared.  Each obstacle is
        # one CBFParams (same gains, distinct centre/radius); the per-tick QP
        # adds one constraint for each of the nearest n_closest_obs of them.
        obs_x = np.array(gp("obstacles_x").value, dtype=float).ravel()
        obs_y = np.array(gp("obstacles_y").value, dtype=float).ravel()
        obs_r = np.array(gp("obstacle_radii").value, dtype=float).ravel()
        if not (obs_x.size == obs_y.size == obs_r.size):
            raise ValueError(
                f"obstacles_x/obstacles_y/obstacle_radii length mismatch: "
                f"{obs_x.size}/{obs_y.size}/{obs_r.size}")
        margin = float(gp("body_margin").value)
        self._n_closest = int(gp("n_closest_obs").value)
        self._gains = dict(
            alpha0=float(gp("alpha0").value),
            alpha_qp=float(gp("alpha_qp").value),
            u_max=self._accel_max,
            qp_penalty=float(gp("qp_penalty").value),
        )
        # self._obstacles: list of (CBFParams (keep-out = real + margin), real_r)
        self._obstacles = [
            (dic.CBFParams(obstacle_xy=(float(cx), float(cy)),
                           obstacle_radius=float(r) + margin, **self._gains),
             float(r))
            for cx, cy, r in zip(obs_x, obs_y, obs_r)
        ]
        if len(self._obstacles) > diagnostics.MAX_OBS:
            raise ValueError(
                f"{len(self._obstacles)} obstacles exceeds diagnostics MAX_OBS="
                f"{diagnostics.MAX_OBS}; raise MAX_OBS in diagnostics.py")
        if not self._obstacles:
            self.get_logger().warn(
                "no obstacles configured -- the CBF will pass the nominal through")
        # Representative CBFParams (shared gains) for building the phi-source;
        # plain/pshr/nmr read the obstacle per-call, so any obstacle works.
        self._cbf = (self._obstacles[0][0] if self._obstacles
                     else dic.CBFParams(obstacle_xy=(0.0, 0.0),
                                        obstacle_radius=0.0, **self._gains))

        # ---- CBF method + noise injection -----------------------------------
        method = str(gp("cbf_method").value).lower().strip()
        self._state_eps = np.array(gp("state_eps").value, dtype=float)
        self._noise_enabled = bool(gp("noise_injection.enabled").value)
        self._noise_bias = np.array(gp("noise_injection.bias").value, dtype=float)
        if self._noise_enabled and np.any(np.abs(self._noise_bias) > self._state_eps):
            self.get_logger().error(
                f"noise_injection.bias {self._noise_bias.tolist()} exceeds "
                f"state_eps {self._state_eps.tolist()}; the CBF cannot guarantee "
                f"anything outside its declared bound.")
        self.get_logger().info(
                f"state_eps {self._state_eps.tolist()}")

        # Resolve the NMR checkpoint path (may live in the share dir or the
        # source tree, matching how the old value_function did it).
        ckpt_param = str(gp("nmrcbf_checkpoint").value)
        ckpt_path = Path(ckpt_param)
        if not ckpt_path.is_absolute():
            share = Path(get_package_share_directory("nick_safe_ctrl"))
            candidates = [share / ckpt_path, share.parents[3] / ckpt_path]
            for c in candidates:
                if c.exists():
                    ckpt_path = c
                    break

        pshr_cfg = dict(
            restarts=int(gp("pshr_restarts").value),
            steps=int(gp("pshr_steps").value),
            lr=float(gp("pshr_lr").value),
            seed=int(gp("pshr_seed").value),
        )
        mr_lo = np.array(gp("mrcbf_lipschitz_lo").value, dtype=float)
        mr_hi = np.array(gp("mrcbf_lipschitz_hi").value, dtype=float)
        mr_samples = int(gp("mrcbf_lipschitz_samples").value)

        self._phi_source = phi_sources.build_phi_source(
            method, self._cbf, self._state_eps,
            pshr_cfg=pshr_cfg,
            nmr_checkpoint=ckpt_path,
            mr_bounds=(mr_lo, mr_hi),
            mr_samples=mr_samples,
        )

        # ---- State -----------------------------------------------------------
        self._latest: Optional[SportModeState] = None
        self._latest_ns: Optional[int] = None
        self._task_origin: Optional[tuple] = None   # (x0, y0, yaw0)
        self._vc = np.zeros(2)                      # integrated control-pt vel
        self._last_tick_ns: Optional[int] = None
        self._stopped = False                       # latched once goal reached
        self._start_checked = False                 # one-time keep-out check

        # ---- ROS plumbing ----------------------------------------------------
        self._sub = self.create_subscription(
            SportModeState, str(gp("state_topic").value), self._on_state, 10)
        self._req_pub = self.create_publisher(
            Request, str(gp("sport_request_topic").value), 10)
        self._diag_pub = self.create_publisher(
            Float64MultiArray, "~/diagnostics", 10)
        self._timer = self.create_timer(1.0 / self._rate, self._on_tick)

        obs_desc = ", ".join(
            f"[{cbf.obstacle_xy[0]:.2f},{cbf.obstacle_xy[1]:.2f}] "
            f"R_cbf={cbf.obstacle_radius:.2f}(={real_r}+{margin})"
            for cbf, real_r in self._obstacles) or "none"
        self.get_logger().info(
            f"safe_ctrl_node up | goal={self._goal.tolist()} | rate={self._rate} Hz | "
            f"control point d={self._d} (NOT added to R_cbf) | "
            f"{len(self._obstacles)} obstacle(s), nearest {self._n_closest} active: "
            f"{obs_desc}")
        self.get_logger().info(
            f"CBF method: {self._phi_source.name} | state_eps={self._state_eps.tolist()}")
        if isinstance(self._phi_source, phi_sources.MRCBFPhi):
            self.get_logger().info(
                f"MR-CBF phi_const = {self._phi_source.phi_const:.4f}")
        if self._noise_enabled:
            self.get_logger().warn(
                f"NOISE INJECTION ENABLED: bias={self._noise_bias.tolist()} "
                f"applied to the CBF state estimate (xhat = state + bias).")
        if self._dry_run:
            self.get_logger().warn(
                "DRY RUN: running the full pipeline and logging commands, "
                "but NOT publishing motion. Set dry_run:=false to drive the robot.")

    # ---- ROS callbacks -------------------------------------------------------

    def _on_state(self, msg: SportModeState) -> None:
        self._latest = msg
        self._latest_ns = self.get_clock().now().nanoseconds

    def _on_tick(self) -> None:
        now = self.get_clock().now().nanoseconds
        if self._latest is None or self._latest_ns is None:
            return

        if (now - self._latest_ns) / 1e9 > self._state_timeout:
            self.get_logger().warn("state timeout -- sending StopMove",
                                   throttle_duration_sec=1.0)
            self._send_stop()
            self._reset_integrator()
            return

        msg = self._latest
        x_odom = float(msg.position[0])
        y_odom = float(msg.position[1])
        yaw_odom = float(msg.imu_state.rpy[2])

        # Capture the task frame on the first good tick.
        if self._task_origin is None:
            self._task_origin = (x_odom, y_odom, yaw_odom)
            self.get_logger().info(
                f"task frame origin captured: odom=({x_odom:.2f}, {y_odom:.2f}), "
                f"yaw={yaw_odom:.2f} rad")

        # Robot pose in the task frame.
        x0, y0, yaw0 = self._task_origin
        c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
        ddx, ddy = x_odom - x0, y_odom - y0
        x = c0 * ddx - s0 * ddy
        y = s0 * ddx + c0 * ddy
        yaw = self._wrap(yaw_odom - yaw0)

        # Integration step from wall-clock between ticks.
        if self._last_tick_ns is None:
            dt = 1.0 / self._rate
        else:
            dt = float(np.clip((now - self._last_tick_ns) / 1e9, 1e-3, 0.1))
        self._last_tick_ns = now

        # Stop and latch once the body reaches the goal.
        goal_dist = math.hypot(x - self._goal[0], y - self._goal[1])
        if self._stopped or goal_dist <= self._goal_tol:
            if not self._stopped:
                self.get_logger().info(
                    f"goal reached (dist={goal_dist:.2f} m) -- stopping")
            self._stopped = True
            self._send_stop()
            self._reset_integrator()
            return

        # Control point from the measured pose; this is the true state.
        xc, yc = control_point(x, y, yaw, self._d)
        true_cbf_state = np.array([xc, self._vc[0], yc, self._vc[1]])

        # Synthesize the noisy estimate xhat = true + bias (when enabled).
        # Both the nominal controller and the CBF QP see only xhat -- matching
        # the notebook's robust-CBF rollout structure.
        if self._noise_enabled:
            cbf_state = true_cbf_state + self._noise_bias
        else:
            cbf_state = true_cbf_state

        # One-time check: the control point must start outside EVERY keep-out
        # zone, or the filter cannot guarantee recovery. We check the TRUE
        # state -- the relevant safety guarantee is on the body, not the
        # noisy estimate.
        if not self._start_checked:
            self._start_checked = True
            for cbf, _ in self._obstacles:
                h0 = dic.barrier_h(true_cbf_state, cbf)
                if h0 > 0.0:
                    self.get_logger().error(
                        f"control point STARTS {h0:.2f} m inside the keep-out of "
                        f"obstacle ({cbf.obstacle_xy[0]:.2f},{cbf.obstacle_xy[1]:.2f}) "
                        f"(R_cbf={cbf.obstacle_radius:.2f} m) -- the filter cannot "
                        f"guarantee recovery. Move the robot or that obstacle apart.")

        # Nominal goal-seeking acceleration on the (noisy) estimate. The nominal
        # targets goal + d*heading so the body (trailing the control point by d)
        # ends at the goal.
        goal_cp = self._goal + self._d * np.array([math.cos(yaw), math.sin(yaw)])
        u_nom = dic.nominal_goto(cbf_state, goal_cp, self._kp, self._kd,
                                 self._accel_max)

        # Take the nearest n_closest_obs obstacles (by the estimate's control-
        # point clearance) as active QP constraints, compute each one's phi, and
        # solve the stacked min-norm CBF-QP. With one obstacle this is identical
        # to the single-constraint closed form.
        active = self._select_obstacles(cbf_state)
        obs_params = [cbf for cbf, _ in active]
        phis = [float(self._phi_source(cbf_state, cbf)) for cbf in obs_params]
        if obs_params:
            u_safe, psi1s, slacks = dic.cbf_qp_filter_multi(
                cbf_state, u_nom, obs_params, phis)
            bind = int(np.argmax(psi1s))          # most-demanding active obstacle
            psi1 = float(psi1s[bind])
            phi = float(phis[bind])
            slack = float(np.max(slacks))
        else:
            u_safe = np.clip(u_nom, -self._accel_max, self._accel_max)
            psi1 = phi = slack = 0.0

        # Integrate acceleration -> control-point velocity; cap the speed.
        self._vc = self._vc + u_safe * dt
        speed = float(np.linalg.norm(self._vc))
        if speed > self._cp_speed_max:
            self._vc *= self._cp_speed_max / speed

        # Feedback-linearize -> unicycle commands; clamp.
        vx, vyaw = to_unicycle_cmd(self._vc[0], self._vc[1], yaw, self._d)
        vx = float(np.clip(vx, self._vx_min, self._vx_max))
        vyaw = float(np.clip(vyaw, -self._vyaw_max, self._vyaw_max))

        # Worst-case barrier over ALL obstacles on the TRUE state, so a noisy
        # run stays interpretable; h_true < 0 => body clear of every keep-out.
        h_trues = [dic.barrier_h(true_cbf_state, cbf) for cbf, _ in self._obstacles]
        h_true = max(h_trues) if h_trues else float("nan")
        self.get_logger().info(
            f"goal={goal_dist:.2f} h_true={h_true:+.3f} psi1={psi1:+.3f} "
            f"phi={phi:.3f} nobs={len(active)}/{len(self._obstacles)} "
            f"u_nom=({u_nom[0]:+.2f},{u_nom[1]:+.2f}) "
            f"u_safe=({u_safe[0]:+.2f},{u_safe[1]:+.2f}) slack={slack:.3f} | "
            f"cmd vx={vx:+.2f} vyaw={vyaw:+.2f}"
            + ("  [dry-run]" if self._dry_run else "")
            + ("  [noisy]" if self._noise_enabled else ""),
            throttle_duration_sec=0.2)

        # Full-rate diagnostics for recording/plotting (published in dry-run
        # too; it is not a motion command). Every obstacle is logged with an
        # active flag = whether it was a QP constraint this tick.
        active_ids = {id(cbf) for cbf in obs_params}
        obs_diag = [
            {"x": cbf.obstacle_xy[0], "y": cbf.obstacle_xy[1],
             "rcbf": cbf.obstacle_radius, "realr": real_r,
             "active": 1.0 if id(cbf) in active_ids else 0.0}
            for cbf, real_r in self._obstacles
        ]
        diag = Float64MultiArray()
        diag.data = diagnostics.pack({
            "t": now / 1e9,
            "x": x, "y": y, "yaw": yaw,
            "xc": xc, "yc": yc,
            "vxc": self._vc[0], "vyc": self._vc[1],
            "xhat_xc": cbf_state[0], "xhat_vxc": cbf_state[1],
            "xhat_yc": cbf_state[2], "xhat_vyc": cbf_state[3],
            "h_true": h_true, "psi1": psi1, "phi": phi,
            "u_nom_x": u_nom[0], "u_nom_y": u_nom[1],
            "u_safe_x": u_safe[0], "u_safe_y": u_safe[1],
            "slack": slack,
            "vx_cmd": vx, "vyaw_cmd": vyaw,
            "goal_dist": goal_dist,
            "goal_x": self._goal[0], "goal_y": self._goal[1],
            "n_obs": len(self._obstacles),
            "n_active": len(active),
        }, obstacles=obs_diag)
        self._diag_pub.publish(diag)

        self._send_move(vx, 0.0, vyaw)

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _wrap(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _select_obstacles(self, cbf_state):
        """Nearest ``n_closest_obs`` obstacles as ``(CBFParams, real_r)`` tuples.

        Ranked by the estimate control point's signed clearance to each keep-out
        (``||p_hat - centre|| - R_cbf``); smaller (closer) ranks first. With
        ``n_closest_obs <= 0`` all obstacles are used.
        """
        if not self._obstacles:
            return []
        px, py = cbf_state[0], cbf_state[2]

        def clearance(item):
            cbf, _ = item
            ox, oy = cbf.obstacle_xy
            return math.hypot(px - ox, py - oy) - cbf.obstacle_radius

        ranked = sorted(self._obstacles, key=clearance)
        n = self._n_closest if self._n_closest > 0 else len(ranked)
        return ranked[:n]

    def _reset_integrator(self) -> None:
        self._vc = np.zeros(2)

    def _send_move(self, vx: float, vy: float, vyaw: float) -> None:
        if self._dry_run:
            return
        req = Request()
        req.header.identity.api_id = SPORT_API_ID_MOVE
        req.parameter = json.dumps({"x": vx, "y": vy, "z": vyaw})
        self._req_pub.publish(req)

    def _send_stop(self) -> None:
        if self._dry_run:
            return
        req = Request()
        req.header.identity.api_id = SPORT_API_ID_STOPMOVE
        self._req_pub.publish(req)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafeCtrlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._send_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
