# Go2 hardware rerun: instructions for the onboard Claude agent

You are working on the Unitree Go2's onboard computer to prepare the software for a rerun of the hardware experiments in the paper "Learning-Based Measurement-Robust Control Barrier Functions for Obstacle Avoidance under State Estimation Error" (RA-L submission 26-5080, in revision).
A human collaborator will run the robot.
Your job is to prepare, verify, and document the software, not to move the robot.

## How the work is split

You are only reachable while the robot is on ethernet. During experiments it is on Wi-Fi, and nobody can reach you. So the work happens in three phases:

| Phase | Connection | Who | What |
|---|---|---|---|
| A | Ethernet | You | Everything in steps 0–7: snapshot, merge, fixes, new weights, bound, simulation, and a runbook |
| B | Wi-Fi | Collaborator alone | One drift check, then every sweep, following your runbook |
| C | Ethernet | You | Analysis, report, push |

**Phase A must leave nothing for the collaborator to work out.** Everything they run in phase B must be one copy-pasteable command from `RUNBOOK.md`. That includes the sweeps, the checks between sweeps, and what to do if a check fails. Use only Python and the tools already onboard. Assume the collaborator cannot edit code or configs during phase B.

## Ground rules

- **Never command the robot.** Do not start `safe_ctrl_node` with `dry_run: false`, call the sport API, or run a sweep script.
- **Snapshot before you change anything.** The code that ran the June 2026 experiments exists only on this computer. It must be preserved exactly before any edit (step 1).
- **Ask before pushing, and never force-push.** Push only to new branches. Show the collaborator the diff summary first.
- **Do not delete or overwrite checkpoints, bags, or configs.** Copy and rename instead.
- When something here does not match what you find onboard, stop and report it rather than guessing.

## Background: what was wrong with the June 2026 runs

The paper's hardware section (Figs. 5–6, Table III) compares a standard CBF against the NMR-CBF over 10 trials each, with two obstacles.
Offline analysis of the June bags (`sweep_plain_20260623_162037`, `sweep_nmrcbf_20260623_163902`) found four problems:

1. **φ was evaluated in the wrong frame.** The NMR network ρ_ξ(x̂, e) was trained with a single obstacle at the origin, so it must be evaluated on the state *relative to each obstacle*. Onboard, `NMRCBFPhi.__call__(xhat, p)` calls `self.net(xhat, self.eps)` with the absolute task-frame estimate and ignores `p`, so both obstacles got the same φ. Replaying the logged states reproduces the logged φ to within 0.005 using the absolute state, versus about 0.18 using obstacle-relative states.
2. **The deployed weights were pretrained only.** `ckpts/phi_params.pkl` (md5 `4abebb89f8bd44e0a57bca88256f2dd0`) matches the supervised target φ* almost exactly. It is replaced for the rerun (step 5).
3. **The error bound was not conservative.** `state_eps` was `[0.05, 0.01, 0.15, 0.01]` (px, vx, py, vy), growing at 0.13 m per metre travelled in px and capped at 0.7. Against Vicon ground truth, the true along-track error grew about 0.2 m per metre, reaching 1.5 m. The lateral error reached about 0.28 m against a 0.15 m bound. The bound was violated on 74–80% of steps in every run.
4. **The bound left the training envelope.** The June network was trained with `EPS_MAX = [0.5, 0.25, 0.5, 0.25]`, but the px bound reached 0.7.

# Phase A (ethernet)

## Step 0: inventory (report to the collaborator before step 1)

1. Find the ROS 2 workspace and the `nick_safe_ctrl` package that actually ran, plus any sweep or launch scripts that produced `run_config.yaml`, `scene.yaml`, `runs.csv`, and `report.json`.
2. For each, report its path, whether it is a git repo, the remote, the branch, `git status`, and the last commit.
3. List every checkpoint under the package's `ckpts/` with its md5. Identify which one the June config used (`nmrcbf_checkpoint: ckpts/phi_params.pkl`).
4. **Vicon does not reach the robot.** It is only available on the lab computer. In June, each run's start pose was read off the lab computer by hand and entered into the sweep tooling, which uses it to place the obstacles (`scene.yaml` is in the Vicon frame; `runs.csv` records `start_x`, `start_y`, `start_yaw_deg`). The ground-truth trajectories were recorded on the lab computer and aligned with the onboard estimate afterwards, off the robot, to produce `aligned_trajectories/<run>/gt.npy` and `est.npy` (columns x, y, yaw). Find exactly how the sweep script takes the start pose (a prompt, an argument, or a file), and report it.
5. Confirm the onboard `phi_sources.py` still has the absolute-frame call described above.

## Step 1: snapshot the onboard code

- **If the onboard package is in a git clone of `robust_pncbf` (or `robust_pncbf-go2`):** create the branch `hardware/onboard-snapshot-2026-06` from the current HEAD. Commit every modified and untracked source, config, and script file. Leave out bags, build artifacts, `__pycache__`, `install/`, `build/`, and `log/`. Use a commit message that says this is the exact code that ran the June 2026 experiments.
- **If it is not a git repo:** clone the repo next to it. Create the same branch from `origin/hardware/go2` (the branch the hardware port started from, commit `7724604`). Copy the onboard files over the matching paths, then commit.
- **Checkpoints:** commit the June checkpoint only if it is small (it is about 20 KB) and not already in the repo under that md5.

After the collaborator approves, push this branch.

## Step 2: merge with the robust-baseline work

Branch `origin/hardware/go2-robust-baselines` (commit `32a5a74`, 2026-10-28) adds R-CBF-QP and Duality CBF methods to `nick_safe_ctrl` (`robust_baselines.py`, `phi_sources.py`, `solve_filter`). It was written against the older repo version, *not* the onboard code.

1. Create `hardware/rerun-r1` from the snapshot branch.
2. Merge `origin/hardware/go2-robust-baselines` into it.
3. Resolve conflicts by **keeping every onboard feature**, including:
   - the `state_eps_schedule` distance schedule;
   - `obstacle_source`, the perception topics, and `perceived_obstacle_radius`;
   - the sweep tooling;
   - the diagnostics layout.

   Add the new methods on top.
4. Make sure `cbf_method` accepts `plain | nmrcbf | rcbfqp | duality`. `pshrcbf` (the DMR-CBF) is optional; see step 6.
5. Run any self-tests (`python double_integrator_cbf.py`, `python phi_net.py`) and a `dry_run: true` launch, to check the node starts with each method.

## Step 3: fix the φ frame

In `phi_sources.py`, the NMR source must evaluate the network on the estimate relative to the obstacle passed in:

```python
def __call__(self, xhat, p):
    ox, oy = p.obstacle_xy
    rel = np.asarray(xhat, dtype=np.float32) - np.array([ox, 0.0, oy, 0.0], dtype=np.float32)
    return self.net(rel, self.eps_now)
```

The state ordering is (px, vx, py, vy); check it against `phi_net.py`. Use the *current* scheduled error bound, not the base `state_eps`; check how the schedule feeds the φ source onboard.

Then check that every other φ source is evaluated in each obstacle's frame. `pshrcbf` and `duality` take `p`; check that they use `p.obstacle_xy`.

Add a test that:
- places two obstacles and checks that the φ values differ and equal the network evaluated at the relative states;
- translates the whole scene (obstacles and state) by a constant offset and checks that every φ is unchanged.

## Step 4: logging needed for the paper

Extend the diagnostics stream, then update `diagnostics.py` and the reader so old bags still decode:

- the current error-bound vector, as fields `eps_px`, `eps_vx`, `eps_py`, `eps_vy`, so the bound is recorded, not reconstructed;
- φ for **every** obstacle slot, as `obs{k}_phi` (NaN when inactive), not just the binding one;
- the per-tick filter compute time in milliseconds, as `filter_ms`, to check that each method keeps up at 50 Hz.

Use exactly these names: `hw_checks.py pilot` looks for them.

**Trial exit.** Every trial must end on its own, with the robot commanded to stop (zero velocity, standing) and the outcome recorded. Add this to the node or the sweep tooling if it is not already there:
- `reached`: the estimate is within `goal_tolerance` of the goal;
- `timeout`: the robot has not reached the goal after 30 s (about 3× the June median), or has been stalled (control-point speed under 0.05 m/s while not at the goal) for 5 s;
- `out_of_bounds`: the estimate leaves the usable space (bounds agreed with the collaborator);
- `aborted`: the collaborator stops the trial by hand (one key in the sweep terminal; the e-stop is always available too).

Make all thresholds config parameters. Record the outcome in `report.json` (`outcome`) and `runs.csv` (`outcome` column), and make the sweep continue to the next trial after any outcome except `aborted`, where it asks whether to continue. Collisions are judged afterwards from ground truth, not by the exit. Test every exit in dry-run mode, including a forced stall.

Every sweep directory must contain a `sweep_meta.json` with keys `git_commit`, `ckpt_md5` and `run_config` (the config path or contents).

**Matching Vicon recordings to runs.** Ground truth is recorded on the lab computer, so every Vicon file must be easy to match to its bag afterwards. Make the sweep script print, at the start of each trial:
- the sweep name and run ID, which the collaborator uses as the Vicon file name;
- the robot's wall-clock time;
- the start pose that was entered.

Write the same three things into `runs.csv`, with the time in a column named `start_wall_time` (`run`, `cbf_method`, `start_x`, `start_y` and `start_yaw_deg` already exist). Do not change how the start pose is entered.

## Step 5: new weights and error bound

**New network.** Deploy `ckpts/phi_params_finetune_e1.2_cap12_unif.pkl` from the folder this file is in (md5 `baf9bc5d0f60b64ecba18384635f23f7`). Copy it into the package's `ckpts/` under the same name, and point `nmrcbf_checkpoint` at it; keep the June checkpoint. It was trained with:
- the obstacle at the origin, so it must be evaluated in each obstacle's frame (step 3);
- error-bound envelope ē = `[1.2, 0.3, 0.5, 0.3]` (px, vx, py, vy), label cap 12;
- `UMAX` = 2.0, `ALPHA_HOCBF` = 3.0, `ALPHA_QP` = 2.0, matching `accel_max`, `alpha0` and `alpha_qp` in `safe_ctrl.yaml` (keep these unchanged);
- supervised pretraining (`pretrain_phi_go2.py`), then fine-tuning with the error bound sampled uniformly over [0, ē] (`finetune_phi_dbint2d_uniform_eps.py`). These scripts need JAX, which is not on this computer; they are included for the record only.

The obstacle radius does not matter: in obstacle-relative coordinates, φ* is independent of the keep-out radius, because the radius enters the drift term only as a constant that cancels. So `body_margin` can stay as it is.

**Course and error.** Keep the June layout. As in June, the along-track (px) error comes only from natural odometry drift: **do not inject any px bias, and do not correct the odometry.** The only injected error stays the −0.15 m py bias.

**Bound.** Set this schedule in the config used for every method; it comes from the June Vicon data:
- px: 0.05 + 0.25 × estimated distance travelled, capped at 1.2. In June the drift was about 19.5% of *true* distance, which is about 0.24 m per metre of *estimated* distance, and the schedule is driven by estimated distance.
- py: 0.30, covering the 0.15 m bias plus June's lateral drift of up to about 0.13 m.
- vx, vy: 0.25, covering the roughly 19% under-reported speed at up to 1 m/s.

In June the along-track error was about 0.5 m at obstacle 1 and 0.9–1.0 m at obstacle 2. The cap is reached around obstacle 2, and the error keeps growing after the last obstacle (to about 1.5 m at the goal). So the bound is exceeded only after the robot has passed both obstacles. Check this on the June data:

```
python hw_checks.py error-bound <june aligned_trajectories dir> --rate 0.25 --base-x 0.05 --cap-x 1.2 --bound-y 0.30
```

`error-bound` applies the rate to distance along the estimate, as the robot does. Report the distance along the course at which the first violation happens in each run, and confirm it is past obstacle 2.

## Step 6: simulation checks

This runs offline on the onboard computer and never commands the robot. It needs only numpy and the onboard package's `double_integrator_cbf.py` and `phi_net.py`, so point it at the *merged* package from step 2:

```
python sim_go2_course.py --pkg-dir <merged>/nick_safe_ctrl/nick_safe_ctrl \
    --new-ckpt ckpts/phi_params_finetune_e1.2_cap12_unif.pkl --old-ckpt <package>/ckpts/phi_params.pkl
```

Run it unchanged first, to confirm you get the table below. Then extend it to every method planned for the robot (`rcbfqp` and `duality` via the merged package's `solve_filter`; `pshrcbf` only as described below). For each, report collisions, minimum clearance, time to goal, lateral range, QP slack use, and compute time per tick.

With the June drift model, Nick got:

| Method | Collisions | Min clearance | Reached | Median time | Lateral range |
|---|---|---|---|---|---|
| CBF | 10/10 | −0.27 m | 10/10 | 9.6 s | −0.6 to +0.6 m |
| June network, evaluated per obstacle, June bound | 10/10 | −0.11 m | 10/10 | 10.6 s | −0.7 to +0.9 m |
| New network (deploy this) | 0/10 | +0.30 m | 10/10 | 11.5 s | −0.8 to +1.0 m |

The simulation is harsher than the robot: it has the CBF colliding 10/10 against 3/10 in June. So only compare methods within it. The June lateral range on hardware was −0.7 to +1.0 m.

**Methods for phase B.** The NMR-CBF must be collision-free and reach the goal in simulation. A baseline is left out of the runbook only if it cannot keep up at 50 Hz, or leaves the usable space (the June lateral range was −0.7 to +1.0 m; ask the collaborator how wide the space actually is). Report why. **Collisions and stalls are not reasons to drop a baseline**: they are the Unsafe and Timeout outcomes the paper already reports in Fig. 2, and the trial exit in step 4 ends stalled trials cleanly. Report which baselines stall or collide in simulation, so the collaborator knows what to expect. For R-CBF-QP in particular, check that its per-tick grid search over (γ₁, γ₂) on states sampled in the error box stays well under 20 ms with the 1.2 m bound, and report whether the large box pushes γ to `gamma_max`.

`pshrcbf` implements the *original* DMR-CBF. The revised paper's constraint is a̲ + b(x̂)u ≥ Ḡ, with both inner optimizations over X̄ ∩ C, where Ḡ = max over s ∈ X̄ ∩ C of Σ_j ū_j |b_j(x̂) − b_j(s)| and ū_j = max(|u̲_j|, |ū_j|). The QP uses an exact penalty, so slack is used only when it is infeasible. Leave it out of phase B unless Nick asks for it and it has been updated to this formulation.

## Step 7: prepare phase B and write `RUNBOOK.md`

Prepare everything the collaborator will run without you, and test every command in dry-run mode.

1. **Configs.** One ready-to-use config per method, identical except for `cbf_method` (and `nmrcbf_checkpoint` for the NMR-CBF). Each has the bound schedule from step 5, the −0.15 m py bias, and the June obstacle positions in `scene.yaml`. Check that `dry_run: true` launches cleanly with each.
2. **Drift check.** A single command that drives one run along the course, with plain nominal control, no obstacles, and *no* injected bias. Vicon is only on the lab computer, so the script must work like this:
   1. take the start pose the same way the sweeps do;
   2. drive the run;
   3. print the onboard estimate's final position in the Vicon frame and the estimated distance travelled;
   4. prompt the collaborator to type in the final position read off the lab computer;
   5. print the along-track and lateral error, with PASS or FAIL.

   The pass rule: along-track error between 15% and 24% of the true distance travelled, and lateral error under 0.15 m. On FAIL, the collaborator stops and contacts Nick.
3. **Sweeps.** One command per method that runs the 10 trials, with the June start positions. In June each method ran as one back-to-back sweep (CBF 16:20, NMR-CBF 16:39). Interleaving keeps battery level and floor wear from lining up with method. So, if the sweep tooling can run a single trial of a chosen method from a chosen start position, give an explicit interleaved order across all methods in the runbook (NMR run 1, CBF run 1, R-CBF-QP run 1, Duality run 1, NMR run 2, …), keeping each method's runs in its own sweep folder. If it can only run whole sweeps, give the sweep order instead, alternating methods in blocks of 5 runs if the tooling allows it.
4. **Pilot (optional, recommended).** A short sweep that tests the whole pipeline before the full sweeps, under its own sweep name (e.g. `pilot_<method>`) so it is never mixed with the real data:
   - one trial of each method planned for phase B, from the middle start position, with Vicon recorded as usual;
   - then a single offline check:

     ```
     python hw_checks.py pilot <pilot sweep dirs> --ckpt <nmr ckpt> --phinet-dir <package>/nick_safe_ctrl
     ```

   This checks that the bags decode, the new fields exist, the logged bound follows the schedule and stays within ē, the filter keeps up at 50 Hz, the NMR-CBF φ is evaluated per obstacle and differs between obstacles, and `runs.csv` and `sweep_meta.json` have what is needed to match the Vicon files. It prints PASS or FAIL per item, then `PILOT PASSED` or `PILOT FAILED`.
   - **Before phase B, run this yourself on a `dry_run: true` pilot recording** (the node publishes diagnostics in dry-run mode). Fix anything that fails. Movement-dependent items, such as the bound growing with distance, may only be fully exercised on the real pilot.
   - In the runbook: if the pilot passes, continue to the full sweeps. If it fails, stop, reconnect over ethernet, and give the output to the agent. The collaborator should also confirm that the pilot's Vicon recordings exist and are named with the printed run IDs.
5. **Checks between sweeps.** After each sweep, these must run offline, with no agent:

   ```
   python hw_checks.py summary <sweep_dir>
   python hw_checks.py phi-frame <sweep_dir> <ckpt>
   ```

   For the NMR-CBF, the phi-frame check must show the *relative* column matching the logged φ. Write in the runbook what the collaborator should see, and what to do if they see something else. For example: stop the NMR-CBF sweep if the relative column does not match, or if more than one trial collides.
6. **`RUNBOOK.md`.** Put it next to this file. Include:
   - a pre-flight list: battery, e-stop, the obstacle positions matching `scene.yaml` (checked with Vicon), the config md5s, and Vicon running on the lab computer;
   - the per-trial routine, step by step: read the start pose on the lab computer and enter it, start the Vicon recording named with the sweep and run ID the script prints, run the trial, stop the recording;
   - the drift check and its pass rule;
   - the pilot, and what to do if it fails;
   - the sweep commands in order, each with the check to run afterwards;
   - where the outputs are written;
   - what to do if something fails.

   Print it for the collaborator before they switch to Wi-Fi.
7. Push `hardware/rerun-r1` (with approval), so the exact code used in phase B is recorded.

# Phase C (ethernet, after the experiments)

1. Read the collaborator's notes and the sweep folders. Run `hw_checks.py summary` and `phi-frame` on every sweep.
2. Once the Vicon trajectories are aligned (`gt.npy` and `est.npy` per run), run `hw_checks.py error-bound` with the step 5 schedule. Report every run where the error left the bound before passing obstacle 2.
3. Write `REPORT.md` in the sweep output folder covering:
   - what changed in the code, and its commit;
   - checkpoint md5s;
   - the bound schedule and the drift check result;
   - the simulation check results;
   - each sweep's summary and the phi-frame check;
   - the error-bound check results;
   - anything unexpected.
4. Bundle the bags, configs, `runs.csv`, `report.json`, aligned trajectories, the drift check, and `REPORT.md` for transfer to Nick. Push any remaining code changes (with approval).

`hw_checks.py` and `sim_go2_course.py` are in the same folder as this file. They need only `numpy` and the onboard `phi_net.py`, `diagnostics.py` and `double_integrator_cbf.py`.
