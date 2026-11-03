# Go2 hardware rerun: collaborator checklist

The rerun replaces the hardware results in the NMR-CBF paper (RA-L 26-5080) and answers a reviewer request for robust baselines and better-justified error bounds.
The onboard Claude agent prepares the software using `AGENT_INSTRUCTIONS.md`.
Because the agent is only reachable over ethernet, the work is in three sessions:

1. **Ethernet: the agent prepares everything** and gives you `RUNBOOK.md`.
2. **Wi-Fi: you run the experiments** from the runbook, without the agent.
3. **Ethernet: the agent analyses** the results and writes the report.

## Before session 1

- [ ] Copy this whole folder (`rebuttal/hardware_rerun/`) onto the onboard computer, for example with `scp -r`. It contains:
  - `AGENT_INSTRUCTIONS.md`, the agent's instructions;
  - `ckpts/`, the new NMR-CBF network;
  - `hw_checks.py` and `sim_go2_course.py`, which the agent runs.

  None of these scripts command the robot.

## Session 1 (ethernet, with the agent)

The robot only needs to be powered up, not standing.

In order:

- [ ] **Start the agent** on the onboard computer, point it at `AGENT_INSTRUCTIONS.md`, and tell it this folder's location.
- [ ] **Approve the snapshot.** Before changing anything, the agent saves the code that ran in June and asks you to approve pushing it as `hardware/onboard-snapshot-2026-06`.
- [ ] **Software work.** The agent then merges in the robust baselines, fixes φ, adds logging, installs the new network and the error bound, adds an automatic exit so stalled trials end cleanly, and asks you anything it can't find out itself: for example, how you enter the start pose, and the usable space for the out-of-bounds stop.
- [ ] **Simulation results.** Look them over. The agent will drop any method that fails in simulation and say why.
- [ ] **Read `RUNBOOK.md`.** The agent writes it last, with every command you'll need in session 2. Read it through, and ask about anything unclear now; the agent won't be available later. In particular, check that:
  - the per-trial routine matches how you'll work: read the start pose on the lab computer and enter it into the sweep script, start a Vicon recording named with the sweep and run ID the script prints, run the trial, stop the recording;
  - it includes the drift check, the pilot, and the sweeps for each method that passed simulation.
- [ ] **Approve the final push** of `hardware/rerun-r1`, which records the exact code you'll run in session 2.

## Session 2 (Wi-Fi, no agent)

Follow `RUNBOOK.md`. In outline:

- [ ] **Layout.** Place the obstacles in the June positions and check them with Vicon against `scene.yaml`: about (−0.96, 0.33) and (1.92, −0.47), radius 0.43 m, goal at (3.5, 0.0). No Vicon recalibration is needed.
- [ ] **One drift check.** A single run along the course without obstacles or injected bias, to confirm the odometry drift is still like June's (about 19.5% of distance travelled). At the end, the script asks you to type in the robot's final position from the lab computer, then prints PASS or FAIL. Record Vicon for this run too. If it fails, stop and contact Nick.
- [ ] **Pilot (optional, recommended).** You run this before collecting the full dataset, to make sure the whole setup works end to end. Run one trial of each method under a separate pilot sweep name, following exactly the routine you'll use for the real trials: enter the start pose, record Vicon, take notes. Then check:
  - [ ] **Software:** run the runbook's single `hw_checks.py pilot` command. It checks the logging, the error bound, per-obstacle φ for the NMR-CBF, and filter timing, and should print `PILOT PASSED`.
  - [ ] **Ground truth:** each pilot trial has a Vicon recording that covers the whole run, named with the printed run ID.
  - [ ] **Behaviour:** watching the trials, the robot started where you entered it, passed the obstacles without contact (except possibly the standard CBF), and stopped near the goal.

  If everything passes, carry on to the full sweeps; the pilot trials are not part of the dataset. If `PILOT FAILED`, or the behaviour looks wrong (e.g. the NMR-CBF stalls or swings very wide), stop, reconnect over ethernet, and show the agent the output and your notes. The whole pilot takes about 10–15 minutes and catches setup problems before they affect 40 trials.
- [ ] **Sweeps.** 10 trials per method, in this priority order if time is short:

  | Priority | Method (`cbf_method`) | Notes |
  |---|---|---|
  | 1 | NMR-CBF (`nmrcbf`) | New network, φ-frame fix, new bound schedule |
  | 2 | Standard CBF (`plain`) | Same as June, for comparison |
  | 3 | R-CBF-QP (`rcbfqp`) | Only if the runbook includes it (see below) |
  | 4 | Duality CBF (`duality`) | Only if the runbook includes it (see below) |

  The agent leaves a baseline out only if, in simulation, it is too slow to run at 50 Hz or swings outside the usable space. A baseline that collides or stalls is still run: those are the Unsafe and Timeout outcomes the paper already reports.

- [ ] **Trials end on their own.** A trial stops the robot automatically when it reaches the goal, after 30 s, after 5 s stalled, or if it leaves the usable space, and records the outcome. If a stalled robot is not stopped automatically, stop it with the abort key or the e-stop, and note it. A stall is a valid result (Timeout); don't nudge the robot or restart the trial.
- [ ] Use the same start positions, py bias and bound schedule for every method; the runbook configs already do this. Follow the runbook's run order. If the sweep tooling allows it, the agent will interleave all the methods trial by trial (NMR run 1, CBF run 1, R-CBF-QP run 1, Duality run 1, NMR run 2, …), so battery level and floor wear don't line up with any one method. Otherwise it gives an order of whole sweeps. Note the order you actually used.
- [ ] After each sweep, run the two check commands in the runbook, which need only Python, and follow its instructions if a check fails.
- [ ] Record for every trial:
  - [ ] the rosbag (written on the robot by the sweep script);
  - [ ] Vicon ground truth for the whole run, recorded on the lab computer and named with the sweep and run ID the script prints;
  - [ ] notes: any physical contact with an obstacle, e-stops or manual interventions, falls, slips, and any trial where the automatic exit did not behave as expected.

## Session 3 (ethernet, with the agent)

- [ ] Align the Vicon recordings with the onboard estimates for every trial and the drift check, the same way as in June (`gt.npy` and `est.npy` per run, as in `aligned_trajectories.zip`). Copy them onto the robot for the agent's checks, or send them to Nick.
- [ ] Give the agent your notes and ask it to run phase C: checks, `REPORT.md`, and the push.
- [ ] Send Nick a bundle containing:
  - the bags, aligned trajectories, `runs.csv`, `report.json`, `scene.yaml` and `run_config.yaml` from every sweep;
  - the drift check and the pilot;
  - your notes;
  - `REPORT.md`.
