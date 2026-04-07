"""
models/tcn_model.py  —  TCN probability model for binary prediction market outcomes

architecture: temporal convolutional network with dilated causal convolutions.
chosen over LSTM/transformer for three reasons specific to this use case:
  1. fixed receptive field: with 4 layers at dilations [1,2,4,8] and kernel=3,
     receptive field = 1 + 2*(3-1)*(1+2+4+8) = 61 ticks — covers the full
     64-tick input window without padding artifacts at sequence boundaries
  2. parallelizable: unlike LSTM, all timesteps computed simultaneously —
     critical for the batched multi-market inference in orchestrator.py
  3. stable gradients: residual connections prevent vanishing gradients across
     the dilated stack, which matters for the long-range momentum features

output head:
  the model outputs three values per sample:
    logit  — raw pre-sigmoid score, used for loss computation (numerical stability)
    p_hat  — sigmoid(logit), the probability estimate fed to the predict SKILL
    sigma  — predicted uncertainty via a separate head. sigma is NOT a true
             bayesian uncertainty — it's a learned scalar that the predict SKILL
             uses as a sanity check flag when it deviates significantly from the
             ensemble. trained implicitly via the brier loss on p_hat.

receptive field calculation:
  each dilated causal conv layer with kernel k and dilation d covers:
    (k - 1) * d timesteps of history
  stacked layers [d=1, d=2, d=4, d=8] with k=3:
    layer 1: (3-1)*1  = 2  ticks
    layer 2: (3-1)*2  = 4  ticks
    layer 3: (3-1)*4  = 8  ticks
    layer 4: (3-1)*8  = 16 ticks
  total receptive field: 1 + 2 + 4 + 8 + 16 = 31 ticks per block
  with 2 blocks: 61 ticks — just covers the 64-tick window
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    causal dilated 1d convolution with left-only padding.

    causal means: output at timestep t depends only on inputs at t' <= t.
    this matches the live inference constraint — we never have future data.
    achieved by padding (kernel-1)*dilation zeros on the LEFT only, then
    trimming the right side of the output to restore the original length.

    dilation expands the receptive field exponentially without increasing
    parameter count — each doubling of dilation doubles coverage at the
    same computational cost.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            dilation    = dilation,
            padding     = 0,   # we handle padding manually for causal masking
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left-pad only: (batch, channels, time) → pad time dim on left
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """
    one residual block: two causal convolutions with the same dilation,
    batch norm, gelu activation, and a residual (skip) connection.

    gelu chosen over relu: smoother gradient near zero reduces dead neuron
    risk in the shallow layers where probability features cluster near 0.5.

    the 1x1 projection conv in the residual path handles the case where
    in_channels != out_channels, keeping the skip connection dimensionally
    consistent without forcing equal channel counts at every layer.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # residual projection: only needed when channel dims differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        return F.gelu(out + residual)


class TCNProbabilityModel(nn.Module):
    """
    full TCN: two blocks of four dilated residual layers each, followed by
    a global average pool and two output heads (probability + uncertainty).

    channel progression: 12 → 32 → 64 → 64 → 64 (within each block)
    the first layer expands from 12 input features to 32 channels to give
    the network enough representational width before the dilated layers.

    global average pool at the end collapses the time dimension:
      (batch, 64, 64) → (batch, 64)
    this makes the model input-length agnostic at inference time — if the
    orchestrator ever feeds a shorter window (e.g. market just opened),
    the model degrades gracefully rather than crashing on a shape mismatch.

    output heads:
      probability head: linear(64 → 1) → logit (no activation)
      uncertainty head: linear(64 → 1) → softplus → sigma ∈ (0, ∞)
        softplus ensures sigma is strictly positive without a hard floor.
    """

    def __init__(
        self,
        n_features:   int = 12,
        n_channels:   int = 64,
        kernel_size:  int = 3,
    ):
        super().__init__()

        # block 1: expand input features to working channel width
        # dilations [1, 2, 4, 8] give receptive field of 31 ticks
        self.block1 = nn.Sequential(
            TCNResidualBlock(n_features,  32,         kernel_size, dilation=1),
            TCNResidualBlock(32,          n_channels, kernel_size, dilation=2),
            TCNResidualBlock(n_channels,  n_channels, kernel_size, dilation=4),
            TCNResidualBlock(n_channels,  n_channels, kernel_size, dilation=8),
        )

        # block 2: same dilation pattern, same channel width
        # second pass extends effective receptive field to ~61 ticks
        self.block2 = nn.Sequential(
            TCNResidualBlock(n_channels, n_channels, kernel_size, dilation=1),
            TCNResidualBlock(n_channels, n_channels, kernel_size, dilation=2),
            TCNResidualBlock(n_channels, n_channels, kernel_size, dilation=4),
            TCNResidualBlock(n_channels, n_channels, kernel_size, dilation=8),
        )

        # collapse time dimension — average across all 64 timesteps
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # probability head: outputs raw logit (sigmoid applied externally)
        self.prob_head = nn.Linear(n_channels, 1)

        # uncertainty head: outputs strictly positive sigma via softplus
        self.sigma_head = nn.Sequential(
            nn.Linear(n_channels, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        kaiming uniform init for conv layers (appropriate for gelu activations).
        zero-init the probability head bias so the model starts near p=0.5,
        avoiding early training instability from large initial logits.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        args:
            x: (batch, 12, 64) float32 tensor from MarketDataPipeline.get_tensor()

        returns:
            p_hat:  (batch, 1) probability in [0, 1] — fed to predict SKILL
            logit:  (batch, 1) raw pre-sigmoid output — used for loss computation
            sigma:  (batch, 1) uncertainty estimate in (0, ∞) — sanity check signal
        """
        out = self.block1(x)    # (batch, 64, 64)
        out = self.block2(out)  # (batch, 64, 64)

        # (batch, 64, 64) → (batch, 64, 1) → (batch, 64)
        out = self.global_avg_pool(out).squeeze(-1)

        logit = self.prob_head(out)          # (batch, 1)
        p_hat = torch.sigmoid(logit)         # (batch, 1)
        sigma = self.sigma_head(out)         # (batch, 1)

        return p_hat, logit, sigma


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("running tcn_model self-test...\n")

    model = TCNProbabilityModel()
    model.eval()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters: {param_count:,}")

    # verify forward pass with contract-compliant input
    x = torch.randn(4, 12, 64).clamp(-5, 5)
    with torch.no_grad():
        p_hat, logit, sigma = model(x)

    assert p_hat.shape  == (4, 1), f"p_hat shape wrong: {p_hat.shape}"
    assert logit.shape  == (4, 1), f"logit shape wrong: {logit.shape}"
    assert sigma.shape  == (4, 1), f"sigma shape wrong: {sigma.shape}"
    assert (p_hat >= 0).all() and (p_hat <= 1).all(), "p_hat outside [0,1]"
    assert (sigma > 0).all(), "sigma must be strictly positive"

    print(f"  input shape:  {x.shape}")
    print(f"  p_hat shape:  {p_hat.shape}  range [{p_hat.min():.3f}, {p_hat.max():.3f}]")
    print(f"  logit shape:  {logit.shape}")
    print(f"  sigma shape:  {sigma.shape}  range [{sigma.min():.3f}, {sigma.max():.3f}]")
    print(f"\nall tcn_model tests passed ✓")