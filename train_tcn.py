"""
train_tcn.py  —  PYTHON SCRIPT (deterministic numerical computation, pytorch)

architecture note: fully standalone training script. imports data_pipeline.py
for feature construction to guarantee zero training/serving skew — the exact
same code path that runs at inference time is used to build training windows.

pipeline:
  parquet files (historical_ingest.py output)
    -> HistoricalDataset (torch Dataset, lazy per-market loading)
    -> MarketDataPipeline.ingest() per tick (same code as orchestrator)
    -> MarketDataPipeline.get_batch_tensor() -> (N, 12, 64) windows
    -> TCNWindowDataset (flat window-level Dataset for DataLoader)
    -> DataLoader (shuffled train, sequential val)
    -> TCNProbabilityModel forward pass
    -> focal loss backward
    -> AdamW + ReduceLROnPlateau
    -> early stopping on val brier score
    -> torch.save() to models/tcn_weights.pt

loss function rationale:
  prediction market outcomes are heavily class-imbalanced in practice.
  a naive BCE loss minimizes by predicting the majority class. focal loss
  addresses this by down-weighting easy correct predictions and focusing
  gradient signal on the confident-but-wrong cases — exactly the failures
  that cause kelly sizing to bet too large and blow up the bankroll.

  focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1 - p)
    gamma=2 (standard): quadratically down-weights confident correct predictions
    alpha=0.80: up-weights the minority class (YES=20.7% in current dataset)

  we also add a confidence penalty term to reinforce calibration:
    calibration_loss = mean((p_model - y)^2)  [this is the brier score itself]
  the composite loss = focal_loss + lambda_cal * calibration_loss
  ensures the model doesn't just learn to discriminate YES/NO but also to
  output probabilities that are numerically correct — critical for kelly sizing.

early stopping:
  patience of 10 epochs on validation brier score (not validation loss).
  brier score is the correct metric here because it measures calibration, not
  just discrimination. a model that outputs p=0.9 for every YES market has
  perfect discrimination but terrible calibration; brier catches this.

  brier_score = mean((p_model - outcome)^2) in [0, 1], lower is better
  random predictor baseline: 0.25 (always output p=0.5)
  good calibration target: < 0.15
  compound skill recalibration threshold: > 0.25 (from SKILL.md)

split strategy:
  market-level 80/20 split — val set contains entirely unseen markets.
  window-level random_split causes leakage: adjacent windows from the same
  market (stride=4) share nearly identical features and trivially predict
  each other. market-level split is the correct evaluation for generalization
  to live unseen markets, which is the actual deployment scenario.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import MarketDataPipeline, OrderBookSnapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ── hardware device routing ────────────────────────────────────────────────────

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("device: MPS (apple silicon)")
    else:
        device = torch.device("cpu")
        logger.info("device: CPU (no gpu detected)")
    return device


# ── dataset ────────────────────────────────────────────────────────────────────

class HistoricalMarketDataset(Dataset):
    """
    preprocesses all parquet files into (window, label) pairs during __init__
    and caches them as numpy arrays. O(1) per __getitem__ during training.

    _market_ids tracks which market each window came from — required for the
    market-level split to correctly assign windows to train/val without leakage.

    accepts an explicit parquet_files list so the caller controls ordering,
    ensuring the split indices computed outside __init__ stay consistent.
    """

    WINDOW_SIZE = 64
    STRIDE      = 4

    def __init__(
        self,
        parquet_dir:   Path,
        max_markets:   Optional[int]  = None,
        parquet_files: Optional[list] = None,
    ):
        self._windows:    list[np.ndarray] = []
        self._labels:     list[float]      = []
        self._market_ids: list[str]        = []  # parallel to _windows — market per window

        if parquet_files is None:
            parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if max_markets:
            parquet_files = parquet_files[:max_markets]

        if not parquet_files:
            raise FileNotFoundError(
                f"no parquet files found in {parquet_dir}. run historical_ingest.py first."
            )

        logger.info(f"preprocessing {len(parquet_files)} markets (stride={self.STRIDE})...")
        skipped = 0

        for parquet_path in parquet_files:
            windows, label = self._process_market(parquet_path)
            if windows is None:
                skipped += 1
                continue
            market_id = parquet_path.stem
            for i in range(windows.shape[0]):
                self._windows.append(windows[i])
                self._labels.append(float(label))
                self._market_ids.append(market_id)

        if not self._windows:
            raise RuntimeError(
                "no valid windows extracted. check min tick count and parquet contents."
            )

        logger.info(
            f"dataset built: {len(self._windows):,} windows from "
            f"{len(parquet_files) - skipped} markets ({skipped} skipped)"
        )
        yes_pct = np.mean(self._labels) * 100
        logger.info(f"class balance: YES={yes_pct:.1f}% NO={100 - yes_pct:.1f}%")

    def _process_market(
        self, parquet_path: Path
    ) -> tuple[Optional[np.ndarray], Optional[int]]:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)
            df    = table.to_pandas()
        except Exception as exc:
            logger.warning(f"failed to read {parquet_path.name}: {exc}")
            return None, None

        if "outcome" not in df.columns:
            logger.warning(f"{parquet_path.name} missing outcome column — skip")
            return None, None

        outcome   = int(df["outcome"].iloc[0])
        market_id = parquet_path.stem
        pipe      = MarketDataPipeline(market_id)

        for _, row in df.iterrows():
            snap = OrderBookSnapshot(
                timestamp        = float(row["timestamp"]),
                best_bid         = float(row["best_bid"]),
                best_ask         = float(row["best_ask"]),
                bid_depth        = float(row["bid_depth"]),
                ask_depth        = float(row["ask_depth"]),
                volume_1h        = float(row["volume_1h"]),
                trade_count      = float(row["trade_count"]),
                last_trade_price = float(row["last_trade_price"]),
                last_trade_size  = float(row["last_trade_size"]),
                expiry_ts        = float(row["expiry_ts"]),
            )
            pipe.ingest(snap)

        batch_tensor = pipe.get_batch_tensor(stride=self.STRIDE)
        if batch_tensor is None:
            return None, None

        return batch_tensor.numpy(), outcome

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._windows[idx].copy())
        y = torch.tensor(self._labels[idx], dtype=torch.float32)
        return x, y


# ── loss function ──────────────────────────────────────────────────────────────

class FocalCalibrationLoss(nn.Module):
    """
    focal(alpha, gamma) + lambda_cal * brier

    focal down-weights easy correct predictions to focus gradient on hard cases.
    brier term enforces numerical calibration — prevents the model from learning
    to discriminate YES/NO correctly while outputting miscalibrated probabilities
    that cause kelly sizing to overbet.
    """

    def __init__(self, alpha: float = 0.80, gamma: float = 2.0, lambda_cal: float = 0.3):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.lambda_cal = lambda_cal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p   = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1.0 - p)

        bce_loss     = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        focal_loss   = (focal_weight * bce_loss).mean()

        calibration_loss = ((p - targets) ** 2).mean()

        return focal_loss + self.lambda_cal * calibration_loss


# ── metric ─────────────────────────────────────────────────────────────────────

def brier_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.sigmoid(logits).cpu()
        y = targets.cpu()
        return float(((p - y) ** 2).mean().item())


# ── training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = resolve_device()

    # ── market-level train/val split ──
    # build the file list once here so both the dataset and the split logic
    # use identical ordering — prevents index misalignment between _market_ids
    # and the val_market_ids set.
    all_files = sorted(Path(args.data_dir).glob("*.parquet"))
    if args.max_markets:
        all_files = all_files[:args.max_markets]

    rng      = np.random.default_rng(args.seed)
    shuffled = all_files.copy()
    rng.shuffle(shuffled)

    n_val_markets  = max(int(len(shuffled) * args.val_split), 1)
    val_market_ids = {f.stem for f in shuffled[:n_val_markets]}
    n_train_markets = len(all_files) - n_val_markets

    logger.info(f"loading dataset from {args.data_dir}...")
    full_dataset = HistoricalMarketDataset(
        Path(args.data_dir),
        max_markets   = args.max_markets,
        parquet_files = all_files,   # pass sorted list — order must match _market_ids
    )

    # assign each window to train or val based on its source market
    train_indices = [
        i for i, mid in enumerate(full_dataset._market_ids)
        if mid not in val_market_ids
    ]
    val_indices = [
        i for i, mid in enumerate(full_dataset._market_ids)
        if mid in val_market_ids
    ]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    logger.info(
        f"market-level split: {len(train_indices):,} train windows / "
        f"{len(val_indices):,} val windows "
        f"({n_train_markets} / {n_val_markets} markets)"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
    )

    # ── model ──
    try:
        from models.tcn_model import TCNProbabilityModel
    except ImportError as exc:
        logger.error(f"cannot import TCNProbabilityModel: {exc}")
        logger.error("ensure models/tcn_model.py exists and defines TCNProbabilityModel")
        sys.exit(1)

    model = TCNProbabilityModel().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"model: {model.__class__.__name__} | trainable params: {param_count:,}")

    # ── loss, optimizer, scheduler ──
    criterion = FocalCalibrationLoss(
        alpha      = args.focal_alpha,
        gamma      = args.focal_gamma,
        lambda_cal = args.lambda_cal,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 5,
        min_lr   = 1e-6,
    )

    # ── early stopping state ──
    best_val_brier   = float("inf")
    best_epoch       = 0
    patience_counter = 0
    best_weights     = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"starting training: epochs={args.epochs} batch={args.batch_size} lr={args.lr}")
    logger.info(f"focal: alpha={args.focal_alpha} gamma={args.focal_gamma} lambda_cal={args.lambda_cal}")
    logger.info(f"early stopping: patience={args.patience} min_delta={args.min_delta}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ── train ──
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            p_hat, logit, sigma = model(batch_x)
            logit = logit.squeeze(-1)

            loss = criterion(logit, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ──
        model.eval()
        val_brier_sum = 0.0
        val_loss_sum  = 0.0
        val_n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                p_hat, logit, sigma = model(batch_x)
                logit = logit.squeeze(-1)

                val_loss_sum  += criterion(logit, batch_y).item()
                val_brier_sum += brier_score(logit, batch_y)
                val_n_batches += 1

        avg_train_loss = np.mean(train_losses)
        avg_val_loss   = val_loss_sum  / max(val_n_batches, 1)
        avg_val_brier  = val_brier_sum / max(val_n_batches, 1)
        current_lr     = optimizer.param_groups[0]["lr"]
        epoch_time     = time.time() - epoch_start

        # step scheduler before logging so lr_tag reflects the reduction
        scheduler.step(avg_val_brier)
        new_lr  = optimizer.param_groups[0]["lr"]
        lr_tag  = " [lr reduced]" if new_lr < current_lr else ""  # ascii-safe, no unicode

        logger.info(
            f"epoch {epoch:>4}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_brier={avg_val_brier:.4f} | "
            f"lr={new_lr:.2e}{lr_tag} | "
            f"time={epoch_time:.1f}s"
        )

        improved = avg_val_brier < best_val_brier - args.min_delta
        if improved:
            best_val_brier   = avg_val_brier
            best_epoch       = epoch
            patience_counter = 0
            best_weights     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  new best val brier: {best_val_brier:.4f} (epoch {best_epoch})")
        else:
            patience_counter += 1
            logger.info(f"  no improvement — patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"early stopping at epoch {epoch} (best epoch was {best_epoch})")
                break

    # ── restore and save ──
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        logger.info(f"restored best weights from epoch {best_epoch}")

    output_path = output_dir / "tcn_weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class":      model.__class__.__name__,
            "best_val_brier":   best_val_brier,
            "best_epoch":       best_epoch,
            "train_args":       vars(args),
            "n_features":       12,
            "window_size":      64,
        },
        output_path,
    )

    logger.info("=" * 60)
    logger.info(f"training complete")
    logger.info(f"  best val brier score: {best_val_brier:.4f} (epoch {best_epoch})")
    logger.info(f"  model saved:          {output_path.resolve()}")
    logger.info(f"  calibration target:   < 0.25 (compound SKILL recalibration threshold)")
    if best_val_brier < 0.25:
        logger.info(f"  status:               PASS - below recalibration threshold")
    else:
        logger.warning(f"  status:               FAIL - above 0.25, review focal params and data")
    logger.info("=" * 60)


# ── cli ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="train TCNProbabilityModel on historical polymarket data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",     type=str,   default="data/historical")
    parser.add_argument("--output-dir",   type=str,   default="models")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split",    type=float, default=0.2)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--min-delta",    type=float, default=5e-4)
    parser.add_argument("--focal-alpha",  type=float, default=0.80)
    parser.add_argument("--focal-gamma",  type=float, default=2.0)
    parser.add_argument("--lambda-cal",   type=float, default=0.3)
    parser.add_argument("--max-markets",  type=int,   default=None)
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    train(args)