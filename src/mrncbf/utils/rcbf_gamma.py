class RCBFGammaState(NamedTuple):
    """Persistent state for the gamma optimizer across timesteps."""
    gamma1: float = 0.001
    gamma2: float = 0.001