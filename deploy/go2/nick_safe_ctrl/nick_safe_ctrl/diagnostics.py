"""Field layout for the safe_ctrl_node diagnostics stream.

The node publishes one ``std_msgs/Float64MultiArray`` per control tick on
``~/diagnostics`` (i.e. ``/safe_ctrl_node/diagnostics``).  Using a plain
std_msgs type keeps the bag portable -- it can be read on any ROS2 machine
without the unitree message packages.  The flat ``data`` array is ordered by
``DIAG_FIELDS``; pair the bag with this module (or copy the list) to decode it.

Multiple obstacles are supported via a **fixed-width** block of ``MAX_OBS``
slots (so the array stays a constant length the plotter can rely on).  ``n_obs``
says how many slots are populated; unused slots are NaN.  Per slot we record the
static geometry (centre, CBF keep-out radius, real radius) plus an ``active``
flag = 1.0 when that obstacle was used as a QP constraint that tick (i.e. it was
among the nearest ``n_closest_obs``), else 0.0.

The scalar safety fields summarise across obstacles: ``h_true`` is the **max**
barrier value over *all* obstacles (the least-safe one; ``< 0`` means the body
is clear of every keep-out), while ``psi1`` / ``phi`` / ``slack`` describe the
binding active obstacle (the one the QP worked hardest on).
"""

MAX_OBS = 8

_BASE_FIELDS = [
    "t",                                            # node clock time, seconds
    "x", "y", "yaw",                                # true body pose (task frame)
    "xc", "yc",                                     # true control point (task frame)
    "vxc", "vyc",                                   # integrated control-point velocity
    "xhat_xc", "xhat_vxc", "xhat_yc", "xhat_vyc",   # noisy CBF state estimate
    "h_true",                                       # max_i barrier_h over ALL obstacles (<0 = safe)
    "psi1",                                         # HOCBF of the binding active obstacle
    "phi",                                          # tightening of the binding active obstacle
    "u_nom_x", "u_nom_y",                           # nominal control-point accel
    "u_safe_x", "u_safe_y",                         # CBF-filtered accel
    "slack",                                        # max QP slack over active constraints
    "vx_cmd", "vyaw_cmd",                           # commanded (vx, vyaw)
    "goal_dist",                                    # body distance to goal
    "goal_x", "goal_y",                             # goal (run-constant)
    "n_obs",                                        # obstacles present (<= MAX_OBS)
    "n_active",                                     # obstacles used as QP constraints this tick
]

# Per-obstacle sub-fields, emitted for each of MAX_OBS slots.
_OBS_SUBFIELDS = ["x", "y", "rcbf", "realr", "active"]


def _obs_field_names():
    names = []
    for k in range(MAX_OBS):
        for sub in _OBS_SUBFIELDS:
            names.append(f"obs{k}_{sub}")
    return names


DIAG_FIELDS = _BASE_FIELDS + _obs_field_names()

_INDEX = {name: i for i, name in enumerate(DIAG_FIELDS)}


def pack(base_values: dict, obstacles=None) -> list:
    """Order base scalars + an obstacle list into the DIAG_FIELDS sequence.

    ``base_values`` must supply every field in ``_BASE_FIELDS``.  ``obstacles``
    is an optional list of dicts (keys ``x, y, rcbf, realr, active``); slots past
    ``len(obstacles)`` (up to ``MAX_OBS``) are filled with NaN.
    """
    missing = set(_BASE_FIELDS) - set(base_values)
    if missing:
        raise KeyError(f"diagnostics missing base fields: {sorted(missing)}")
    vals = {name: float(base_values[name]) for name in _BASE_FIELDS}

    obstacles = obstacles or []
    if len(obstacles) > MAX_OBS:
        raise ValueError(f"{len(obstacles)} obstacles exceeds MAX_OBS={MAX_OBS}")
    for k in range(MAX_OBS):
        if k < len(obstacles):
            o = obstacles[k]
            for sub in _OBS_SUBFIELDS:
                vals[f"obs{k}_{sub}"] = float(o.get(sub, float("nan")))
        else:
            for sub in _OBS_SUBFIELDS:
                vals[f"obs{k}_{sub}"] = float("nan")

    return [vals[name] for name in DIAG_FIELDS]


def column(name: str):
    """Index of a field in the diagnostics array."""
    return _INDEX[name]
