import jax
import jax.numpy as jnp
import flax.linen as nn


class ResidualCorrection(nn.Module):
    """
    Learned residual Δ_θ that corrects the analytical robustifying term.
 
    Outputs (nh,) — one correction per CBF constraint, matching the
    per-constraint structure in rcbf_qp_linear_mats.
    """
    nh: int                     # Number of CBF constraints (task.nh)
    hidden_dim: int = 64
    gru_dim: int = 64
    use_temporal: bool = True
    Delta_max: float = 3.0
 
    @nn.compact
    def __call__(self, x_hat, h_Lgh_norm, h_val, epsilon, history_seq=None):
        """
        Args:
            x_hat:        (nx,) current state estimate
            h_Lgh_norm:   (nh,) per-constraint ‖L_g h_i(x̂)‖
            h_val:        (nh,) per-constraint h_i(x̂)
            epsilon:      scalar, reported uncertainty bound
            history_seq:  (window_len, feat_dim) or None
                          feat_dim = nx + 1 + nh  (x̂, ε, ‖Lgh‖ per constraint)
        Returns:
            h_Delta: (nh,) signed residual per constraint
        """
        # ---- Current-state features ----
        features = jnp.concatenate([
            x_hat,                          # (nx,)
            h_Lgh_norm,                     # (nh,)
            h_val,                          # (nh,)
            jnp.array([epsilon]),           # (1,)
        ])
 
        x = nn.Dense(self.hidden_dim)(features)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        state_embedding = x  # (hidden_dim,)
 
        # ---- Temporal features ----
        if self.use_temporal and history_seq is not None:
            # Must use nn.scan (Flax's lifted scan), NOT jax.lax.scan,
            # because Flax modules can't be called inside raw JAX transforms.
            ScanGRU = nn.scan(
                nn.GRUCell,
                variable_broadcast='params',
                split_rngs={'params': False},
                in_axes=0,      # scan over axis 0 of the input (time)
                out_axes=0,
            )
            carry = jnp.zeros(self.gru_dim)
            carry, _ = ScanGRU(features=self.gru_dim)(carry, history_seq)
            temporal_embedding = carry  # (gru_dim,) — final hidden state
 
            combined = jnp.concatenate([state_embedding, temporal_embedding])
        else:
            combined = state_embedding
 
        # ---- Per-constraint output ----
        x = nn.Dense(self.hidden_dim // 2)(combined)
        x = nn.relu(x)
        raw = nn.Dense(self.nh)(x)  # (nh,)
 
        # Bounded output via tanh
        h_Delta = self.Delta_max * nn.tanh(raw)
        return h_Delta