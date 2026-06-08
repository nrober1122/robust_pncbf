# nick_safe_ctrl — Measurement-Robust CBF Obstacle Avoidance on the Unitree Go2

Runs control-barrier-function (CBF) obstacle avoidance on a Unitree Go2, with
four interchangeable barrier-tightening strategies for robustness to
measurement noise. The robot is modeled as a 2-D double integrator on a
feedback-linearization control point; the resulting acceleration is converted
to `(vx, vyaw)` and sent to the Go2's sport-mode `Move` API.

The math is a hardware port of the research notebooks in this package:
`dbint2d_cbf_compare.ipynb`, `dbint2d_quad.py` (CBF + dynamics), and
`linearized.ipynb` (feedback-linearization, unicycle branch).

---

## 1. What it does

Each 50 Hz control tick:

1. Read the robot pose `(x, y, yaw)` from `SportModeState` (onboard odometry +
   IMU). On the first tick this pose defines a **task frame** (origin =
   startup position, +x = startup heading).
2. Place a **control point** `d = 0.5 m` ahead of the robot. The control point
   obeys holonomic double-integrator dynamics under feedback linearization, so
   the CBF runs on it (state `(xc, vxc, yc, vyc)`).
3. (Optional) Add a synthetic **measurement bias** to form a noisy estimate
   `xhat = state + bias`, to test the measurement-robust CBF.
4. Compute a goal-seeking nominal acceleration, then filter it through a
   min-norm **CBF-QP** with a barrier-tightening term `phi >= 0` from the
   selected method.
5. Integrate the safe acceleration into a control-point velocity, cap the
   speed, and map it through the linearizing transform to `(vx, vyaw)`.
6. Publish a sport-mode `Move` request on `/api/sport/request`.

### Multiple obstacles

Obstacles are configured as parallel arrays (`obstacles_x`, `obstacles_y`,
`obstacle_radii`; see §5). Each tick the controller ranks them by the control
point's clearance and adds the **nearest `n_closest_obs`** as separate CBF-QP
constraints (one row per obstacle). The QP then finds the single min-norm
acceleration that satisfies all active constraints at once — so the robot
naturally threads between obstacles rather than handling them one at a time.

With one obstacle (or `n_closest_obs = 1`) the multi-constraint solver reduces
*exactly* to the original single-constraint closed form, so single-obstacle
behaviour is unchanged. The solver is projected Gauss-Seidel on the QP dual
(`double_integrator_cbf.cbf_qp_filter_multi`); still numpy-only, no QP library.

The CBF protects the **control point**, which rides `d` ahead of the body.
Each keep-out radius is `R_cbf = obstacle_radius + body_margin` — the control
point distance `d` is **deliberately not added**. Guarding the control point
rather than the body leaves up to `d` of body slack in a worst-case head-on
approach, but in practice the body swerves with the control point, so that
margin is traded away to let the robot reach closer to obstacles.

### The four CBF methods (`cbf_method` in YAML)

| method | `phi` source | notes |
|---|---|---|
| `plain` | 0 | baseline, no robustness |
| `mrcbf` | constant Lipschitz tightening | **too conservative** — in sim it overshoots so much the goal is missed; kept only as a reference |
| `pshrcbf` | runtime projected-gradient ascent over `[xhat-eps, xhat+eps]` | the worst-case "phi-oracle" |
| `nmrcbf` | learned `PhiNet(xhat, eps)` | numpy port of the trained Flax net; approximates `pshrcbf` |

All four share the same QP; they differ only in how `phi` is computed. With
multiple obstacles, `phi` is computed per active obstacle.

---

## 2. Module map

```
nick_safe_ctrl/
  safe_ctrl_node.py        ROS2 node: state -> task frame -> control point ->
                           noise -> nominal -> CBF-QP -> feedback lin. -> Move
  double_integrator_cbf.py HOCBF psi1 + min-norm CBF-QP; single-constraint
                           closed form + multi-obstacle dual solver (no autodiff)
  feedback_linearization.py control point + linearizing map -> (vx, vyaw)
  phi_sources.py           PlainPhi / MRCBFPhi / PSHRPhi / NMRCBFPhi + dispatch
  phi_net.py               numpy PhiNet + Flax-pickle loader (no jax dep)
  compare_methods.py       offline 4-method comparison rollout
  run_analysis.py          analyze a diag bag: collisions, clearances, goal
  plot_run.py              render a diag bag (or overlay several) to a PNG
  value_function.py        LEGACY/UNUSED (old jax-based loader; superseded)
config/safe_ctrl.yaml      all parameters
launch/safe_ctrl.launch.py launches the node with the YAML
scripts/record_experiment.sh  prompt -> snapshot config -> record bag -> report
ckpts/phi_params.pkl       trained PhiNet weights (used by nmrcbf)
```

The node uses **numpy + rclpy only** — no jax/flax needed on the robot.

---

## 3. Network setup (read this first)

The Jetson has **one wired port (`eth0`)** which must serve EITHER internet
OR the robot — not both at once. The Go2's internal network is
`192.168.123.0/24` (robot MCU at `192.168.123.161`).

- **Robot mode** — NM profile **`go2-ros2`**: `eth0` static `192.168.123.18/24`.
  CycloneDDS is bound to `eth0` (`/home/unitree/cyclonedds_ws/cyclonedds.xml`),
  so ROS sees the Go2's topics only in this mode.
- **Internet mode** — NM profile **`Profile 1`**: `eth0` DHCP. Needed for
  internet on the Jetson (e.g. to run Claude Code on it).

Switch between them:
```bash
sudo nmcli connection up go2-ros2      # robot net (ROS topics appear)
sudo nmcli connection up "Profile 1"   # internet (ROS topics disappear)
```

**SSH access** is over the Jetson's own WiFi access point (it broadcasts
SSID **`DroneBlocks-Go2-001`** via the USB Realtek dongle; NM profile
`Go2-Hotspot`, `shared` mode at **`10.42.0.1`**). Connect your laptop to that
SSID and `ssh unitree@10.42.0.1`. This AP is on `10.42.0.x` specifically so it
does **not** collide with the robot's `192.168.123.x` net — do not move it back
onto `192.168.123.x`.

Verify the robot is reachable in robot mode:
```bash
ping -c3 192.168.123.161
ros2 topic list | grep sportmodestate   # should show /lf/sportmodestate
```

---

## 4. Build & environment

ROS2 **Foxy**. Source order (the unitree messages live in `ros2_ws`):
```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/ros2_ws/install/setup.bash
source /home/unitree/nmr-cbf_ws/install/setup.bash
```

Build from the **workspace root** (`colcon` resolves the package layout relative
to where you invoke it — building from inside `nick_safe_ctrl/` will not work):
```bash
cd /home/unitree/nmr-cbf_ws
colcon build --symlink-install                              # whole workspace
colcon build --packages-select nick_safe_ctrl --symlink-install   # just this pkg (faster)
```
Then source the overlay (only in shells that don't already have it):
```bash
source install/setup.bash
```

> **Always pass `--symlink-install`.** It is what makes edits to
> `config/safe_ctrl.yaml` and the `.py` sources take effect on the **next launch
> with no rebuild and no re-source** — `--symlink-install` symlinks the installed
> files back to the source tree (`install/ → build/ → source`) instead of copying
> them. A bare `colcon build` (no flag) silently replaces those symlinks with
> frozen copies, after which `ros2 launch` keeps loading the stale copy and your
> source edits are ignored until you rebuild. If a run ever behaves as though it
> is ignoring your YAML edits, check this first:
> ```bash
> ls -l install/nick_safe_ctrl/share/nick_safe_ctrl/config/safe_ctrl.yaml
> ```
> It should be a symlink (`-> .../build/.../safe_ctrl.yaml`), not a plain file.
> Re-run `colcon build --packages-select nick_safe_ctrl --symlink-install` to fix.

You rarely need to rebuild at all for `nick_safe_ctrl`: thanks to the symlinks,
editing the `.py` sources or `config/safe_ctrl.yaml` applies on the next
`ros2 launch` directly. A rebuild is only required when you **add a new file**,
edit `setup.py`/`package.xml`, or add a new entry point.

---

## 5. Running an experiment

### Pre-flight
- Robot **standing** (BalanceStand) — `Move` is ignored when sitting/damped.
- Pointed at ~4 m of clear physical space (the CBF only avoids the *virtual*
  obstacle; it does not see real walls). Startup heading = task-frame +x.
- On the `go2-ros2` network (`ros2 topic list` shows `/lf/sportmodestate`).

### Always dry-run first
Set `dry_run: true` in the YAML, then:
```bash
ros2 launch nick_safe_ctrl safe_ctrl.launch.py > ~/sc_dry.log 2>&1
```
Watch (in another terminal): `tail -f ~/sc_dry.log`. The full pipeline runs and
logs the `(vx, vyaw)` it *would* send, but commands no motion. Note: in dry-run
the robot doesn't move, so the logged values plateau after ~1 s (the internal
velocity saturates while the measured pose is fixed) — this is expected.

### Go live
Set `dry_run: false`, relaunch. Expected: robot walks toward the goal, swerves
around the virtual obstacle, and stops when the body reaches the goal
(`goal reached -- stopping` in the log). Ctrl-C also sends `StopMove`.

### Emergency stop (second terminal)
```bash
ros2 topic pub --once /api/sport/request unitree_api/msg/Request \
  '{header: {identity: {api_id: 1003}}}'   # api_id 1003 = StopMove
```

### Key parameters (`config/safe_ctrl.yaml`)
- `cbf_method`: `plain | mrcbf | pshrcbf | nmrcbf`
- `noise_injection.enabled` / `.bias`: synthetic measurement bias (4-vector on
  `(px, vx, py, vy)`); `|bias_i|` must be `<= state_eps_i`
- `state_eps`: uncertainty bound the robust methods assume (default
  `[0.10, 0.05, 0.10, 0.05]`)
- `obstacles_x`, `obstacles_y`, `obstacle_radii`: parallel arrays defining the
  obstacles (one entry each; same length). `body_margin`: shared extra clearance.
  `n_closest_obs`: how many nearest obstacles become QP constraints each tick
  (`<= 0` = all). `goal_xy`, `goal_tolerance`: task-frame goal. Each obstacle
  must be at least `R_cbf + d` from the start (the control point starts `d`
  ahead) or it begins inside that keep-out (the node logs an error if so).
- limits: `accel_max`, `cp_speed_max`, `vx_min/max`, `vyaw_max`

### Reading the per-tick log
```
goal=3.60 h_true=-0.540 psi1=-0.733 phi=0.491 nobs=2/2 u_nom=(+0.50,+0.05)
u_safe=(+0.05,-0.06) slack=0.002 | cmd vx=+0.24 vyaw=-0.64 [noisy]
```
- `goal` — body distance to goal (m)
- `h_true` — **true** body barrier value, **max over all obstacles** (`<0` =
  body safely outside *every* keep-out)
- `psi1` — HOCBF value of the binding (most-demanding) active obstacle
- `phi` — barrier tightening of that obstacle (`>0` = robustness active)
- `nobs` — `active / total` obstacles (active = used as QP constraints this tick)
- `u_nom` / `u_safe` — nominal vs CBF-filtered control-point acceleration
- `cmd` — the `(vx, vyaw)` actually commanded

---

## 6. Offline validation (no robot, no ROS)

```bash
cd nick_safe_ctrl/nick_safe_ctrl
python3 double_integrator_cbf.py    # single- + 2-obstacle rollouts; never penetrated
python3 feedback_linearization.py   # closed-loop unicycle; clears obstacle + reaches goal
python3 phi_net.py                  # loads the checkpoint, prints sample phi values
python3 compare_methods.py          # all 4 methods under a shared bias, side by side
```

---

## 7. Recording & visualizing data

The node publishes a full-rate (50 Hz) diagnostics stream on
**`/safe_ctrl_node/diagnostics`** (`std_msgs/Float64MultiArray`) carrying
everything needed to reconstruct a run: true body pose, the noisy estimate,
`h_true`, `psi1`, `phi`, `u_nom`/`u_safe`, the `(vx, vyaw)` commands, and the
run geometry. Multiple obstacles are recorded in a **fixed-width block** of
`MAX_OBS` (=8) slots — each slot has the obstacle's centre, `R_cbf`, real
radius, and an `active` flag (1 if it was a QP constraint that tick); `n_obs`
gives how many are populated, the rest are NaN. This keeps the array a constant
length the plotter can rely on. The field order is defined in
`nick_safe_ctrl/diagnostics.py` (`DIAG_FIELDS`). It publishes in dry-run too.

### Recommended: `record_experiment.sh` (one command per run)

For a series of experiments, use the wrapper instead of recording by hand. It
asks for a name + notes, snapshots the config **as it was for that run**, records
the bag, and auto-analyzes it for collisions when you stop:

```bash
# 1. launch the node in another terminal (real or dry-run), then:
nick_safe_ctrl/scripts/record_experiment.sh
#    Experiment name: pshr_two_obs
#    Notes (optional): obstacles at 2.84/4.5, bias on y
#    ... records ... press Ctrl-C to stop ...
```

Each run lands in its own timestamped folder under `~/bags/` (override with
`BAG_ROOT=/dir`):

```
~/bags/20260526_141133_pshr_two_obs/
  bag/              the rosbag2 recording
  safe_ctrl.yaml    copy of the source config at record time
  params_live.yaml  the node's ACTUAL loaded params (ros2 param dump) -- the
                    authoritative record of what ran, incl. any non-YAML defaults
  meta.txt          name, timestamp, notes, topics, host
  report.txt        collision / clearance / keep-out / goal analysis (also .json)
  plot.png          3-panel plot (trajectory + obstacles, h_true/phi, vx/vyaw)
```

`params_live.yaml` is captured only if the node is running when you start the
recorder (it is — you launched it first), and is the ground truth for the config
version, since it reflects exactly what the node loaded rather than what the file
happens to say now. To record the raw robot topics too, edit the `TOPICS` array
near the top of the script.

### Analyzing a run for collisions

`record_experiment.sh` runs this automatically, but you can re-run it on any bag:

```bash
python3 nick_safe_ctrl/run_analysis.py ~/bags/<run>/bag [--body-radius 0.25] [--json out.json]
```
It reports, per obstacle, the **minimum body-centre → obstacle-surface
clearance** (using the obstacle's *real* radius, not the inflated `R_cbf`) and
flags a **collision** if the body centre ever came within that radius — with the
time and position of closest approach. It also flags keep-out entries
(`h_true > 0`) and whether the goal was reached. `--body-radius R` inflates the
body to account for the robot footprint (default 0 = point). Exit code is `1` on
any collision, so scripts can branch on it.

### Manual recording (if you prefer)

```bash
ros2 bag record -o ~/bags/<name> /safe_ctrl_node/diagnostics
```
Using only the std_msgs diag topic keeps the bag portable — it reads on any
ROS2 machine without the unitree message packages. (Add `/lf/sportmodestate`
and `/api/sport/request` to the record list if you also want the raw robot
data.)

**Plot a run, or overlay several** (e.g. a plain-vs-pshrcbf A/B comparison):
```bash
python3 nick_safe_ctrl/plot_run.py ~/bags/<name> --out run.png
python3 nick_safe_ctrl/plot_run.py ~/bags/plain ~/bags/pshr --labels plain pshr --out ab.png
```
Produces a 3-panel PNG — XY trajectory (with every obstacle + keep-out circle
and the goal), `h_true`/`phi` vs time, `vx`/`vyaw` vs time — and prints the
**minimum body→surface clearance for each obstacle and the worst-case over
all**, the key safety metric for comparisons.

The plotter reads the bag via `rosbag2_py` if present, else parses the sqlite3
`.db3` directly (only needs `rclpy` + `std_msgs`). `rosbag2_py` is **not**
installed on this Jetson, but the `ros2 bag record` CLI and the sqlite fallback
both work here; plotting also works on a laptop with ROS2 + matplotlib.

---

## 8. Status & known limitations

**Validated so far:** plain-CBF live run (robot avoids virtual obstacle,
reaches goal); PSHR-CBF + noise dry-run (phi active, body clear, more
conservative commands than plain). PSHR-CBF live and the plain-vs-PSHR A/B
comparison are the current next steps.

**Limitations / future work:**
- Ground truth is the Go2's onboard odometry, not Vicon. Fine for short runs;
  for rigorous robustness claims, integrate Vicon (currently on a separate WiFi).
- Obstacles are **virtual** (configured in YAML). Real-obstacle detection from
  the Go2's lidar / `range_obstacle` sensors is a separate, larger task.
- `mrcbf` is too conservative to be useful. `R-CBF` and `R-CBF-QP` from the
  notebook work better and are reasonable to port (different QP structure —
  sampled multi-constraint) but are not yet implemented.
- To re-enable the learned net, set `cbf_method: nmrcbf`. The numpy `PhiNet`
  loads `ckpts/phi_params.pkl` directly (no jax needed).
