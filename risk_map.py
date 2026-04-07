"""
models/risk_map.py  —  PYTHON SCRIPT (deterministic)

architecture note: the TRANSFER FUNCTION (signal → position size) is
deterministic math and must be a python script. the POLICY DECISION
(whether to override the sizing based on market context) is in the skill layer.

the risk map pipeline:
  raw signal s ∈ (-1, +1)
    → deadband filter  (|s| < DEADBAND → 0)
    → volatility scaling  (divide by σ_p̂ to shrink size under uncertainty)
    → clip to [W_MIN, W_MAX]
    → final position weight w

Time complexity:  O(B) where B = batch size (vectorized ops)
Space complexity: O(B)
"""

import torch
import torch.nn as nn
from torch import Tensor

from config.settings import MODEL


def probability_to_signal(p_hat: Tensor) -> Tensor:
    """
    maps calibrated probability p̂ ∈ (0,1) to a signed signal s ∈ (-1, +1).
    formula: s = 2 * p̂ - 1
    interpretation:
        p̂ = 0.9 → s = +0.8  (strong long signal)
        p̂ = 0.5 → s =  0.0  (no signal, within deadband)
        p̂ = 0.2 → s = -0.6  (short signal)

    Time: O(B)  Space: O(B)
    """
    return 2.0 * p_hat - 1.0


def apply_deadband(signal: Tensor, deadband: float) -> Tensor:
    """
    neutral zone filter: signals with |s| < deadband → 0.
    prevents overtrading on marginal signals near 50/50.
    equivalent to a threshold on predicted edge before sizing.

    Time: O(B)  Space: O(1) (in-place-compatible)
    """
    return torch.where(signal.abs() < deadband, torch.zeros_like(signal), signal)


def apply_volatility_scaling(signal: Tensor, sigma: Tensor, eps: float = 1e-6) -> Tensor:
    """
    scales signal inversely proportional to forecast uncertainty.
    when the model is uncertain (high σ_p̂), position size shrinks.
    when the model is confident (low σ_p̂), position size expands.

    formula: scaled_s = s / (σ_p̂ + ε)
    note: σ_p̂ is variance from monte carlo dropout, not price volatility.

    the ε floor prevents division by zero for deterministic (zero-variance) signals.

    Time: O(B)  Space: O(B)
    """
    return signal / (sigma.sqrt() + eps)     # σ → std dev before dividing


class RiskMapTransferFunction(nn.Module):
    """
    converts model probability output to a final position weight.
    this is a deterministic, parameter-free module (no learnable weights).
    it enforces the deadband, volatility scaling, and position bounds
    as hard structural constraints outside the neural network.

    the separation of sizing logic from the neural network means:
    - risk constraints can be changed without retraining the model
    - position sizing is auditable and interpretable
    - model and risk can be unit tested independently
    """

    def __init__(
        self,
        w_min: float    = MODEL.W_MIN,
        w_max: float    = MODEL.W_MAX,
        deadband: float = MODEL.DEADBAND,
    ):
        super().__init__()
        self.w_min    = w_min
        self.w_max    = w_max
        self.deadband = deadband

    def forward(self, p_hat: Tensor, sigma_p_hat: Tensor) -> Tensor:
        """
        full transfer function pipeline.

        args:
            p_hat:       calibrated probability  (B, 1)
            sigma_p_hat: forecast variance       (B, 1)

        returns:
            w: final position weight in [W_MIN, W_MAX]  (B, 1)

        pipeline stages are applied sequentially so each stage is testable:
        s = signal(p_hat)           → raw directional signal in (-1, +1)
        s = deadband(s)             → zero out weak signals
        s = vol_scale(s, sigma)     → reduce size under uncertainty
        w = clip(s, w_min, w_max)   → hard bounds on final exposure
        """

        # stage 1: probability → directional signal
        s = probability_to_signal(p_hat)                 # ∈ (-1, +1)

        # stage 2: deadband — suppress noise near 50/50
        s = apply_deadband(s, self.deadband)

        # stage 3: uncertainty scaling — larger σ → smaller position
        # this is the key coupling between the TCN uncertainty estimate and sizing
        s = apply_volatility_scaling(s, sigma_p_hat)

        # stage 4: clip to hard position bounds
        w = torch.clamp(s, self.w_min, self.w_max)

        return w

    def explain(self, p_hat: float, sigma_p_hat: float) -> dict:
        """
        scalar version for logging and debugging. returns intermediate values.
        Time: O(1)  Space: O(1)
        """
        p_t   = torch.tensor([[p_hat]])
        sig_t = torch.tensor([[sigma_p_hat]])

        s_raw  = probability_to_signal(p_t).item()
        s_db   = apply_deadband(torch.tensor([[s_raw]]), self.deadband).item()
        s_vs   = apply_volatility_scaling(
            torch.tensor([[s_db]]), sig_t
        ).item()
        w_final = max(self.w_min, min(self.w_max, s_vs))

        return {
            "p_hat":          p_hat,
            "sigma_p_hat":    sigma_p_hat,
            "raw_signal":     s_raw,
            "after_deadband": s_db,
            "after_vol_scale":s_vs,
            "final_weight":   w_final,
            "deadband_active": abs(s_raw) < self.deadband,
        }


# ── self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    risk_map = RiskMapTransferFunction()

    # test 1: strong signal, low uncertainty
    r = risk_map.explain(p_hat=0.80, sigma_p_hat=0.001)
    print(f"strong signal: {r}")
    assert r["final_weight"] > 0, "strong long signal should give positive weight"

    # test 2: weak signal within deadband
    r = risk_map.explain(p_hat=0.54, sigma_p_hat=0.001)
    print(f"deadband signal: {r}")
    assert r["after_deadband"] == 0.0, "weak signal should be zeroed by deadband"

    # test 3: strong signal, high uncertainty → should be reduced
    r_low_unc  = risk_map.explain(p_hat=0.80, sigma_p_hat=0.001)
    r_high_unc = risk_map.explain(p_hat=0.80, sigma_p_hat=0.100)
    assert abs(r_low_unc["final_weight"]) >= abs(r_high_unc["final_weight"]), \
        "higher uncertainty should reduce position size"
    print(f"uncertainty scaling: low_unc={r_low_unc['final_weight']:.4f}, "
          f"high_unc={r_high_unc['final_weight']:.4f}")

    print("all risk map assertions passed ✓")
