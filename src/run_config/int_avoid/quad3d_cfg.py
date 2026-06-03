from mrncbf.dyn.quad3d import Quad3D
from mrncbf.pncbf.pncbf import PNCBFCfg, PNCBFEvalCfg, PNCBFTrainCfg
from mrncbf.utils.schedules import Constant, Lin, SchedCtsHorizon
from run_config.loop_cfg import LoopCfg
from run_config.run_cfg import RunCfg


def get(seed: int) -> RunCfg[PNCBFCfg, LoopCfg]:
    dt = Quad3D.DT

    # Horizon schedule: ramp from 10 steps (~0.5 s) to 200 steps (~10 s).
    sched1_steps  = 150_000
    sched1_warmup =  30_000
    lam = SchedCtsHorizon(Lin(10, 200, sched1_steps, warmup=sched1_warmup), dt)

    lr = Constant(3e-4)
    wd = Constant(4e-3)

    # Gradually introduce the bootstrap term to stabilise early training.
    tgt_rhs = Lin(0.0, 0.9, steps=sched1_steps, warmup=sched1_warmup)

    collect_size = 8192
    train_cfg = PNCBFTrainCfg(
        collect_size,
        rollout_dt=dt,
        rollout_T=48,           # 2.4 s rollouts
        batch_size=8192,
        lam=lam,
        tau=0.005,
        tgt_rhs=tgt_rhs,
    )
    eval_cfg = PNCBFEvalCfg(eval_rollout_T=80)
    alg_cfg = PNCBFCfg(
        act="tanh",
        lr=lr,
        wd=wd,
        hids=[256, 256, 256],   # one fewer layer than F16 (simpler system)
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_Vs=2,
        n_min_tgt=2,
    )
    loop_cfg = LoopCfg(
        n_iters=lam.total_steps + 20_000,
        ckpt_every=5_000,
        log_every=100,
        eval_every=5_000,
    )
    return RunCfg(seed, alg_cfg, loop_cfg)
