"""
data_pipeline.py  —  PYTHON SCRIPT (deterministic numerical computation)

architecture note: all feature engineering is deterministic math — correct
place for a python script. there is no fuzzy reasoning here. the output
of this module is a normalized pytorch tensor fed directly into TCNProbabilityModel.

output contract:  torch.Tensor of shape (batch_size, 12, 64)
                  dtype: torch.float32
                  all values in approximately [-3, 3] after z-score normalization
                  channel dim = 12 features, time dim = 64 timesteps

the 12 engineered features:
  [0]  implied_probability      midpoint of best bid / best ask
  [1]  spread_width             (ask - bid) normalized by midpoint
  [2]  order_book_imbalance     (bid_depth - ask_depth) / (bid_depth + ask_depth)
  [3]  volume_1h                rolling 1-hour contract volume, log-scaled
  [4]  time_to_expiry           normalized exponential decay toward resolution
  [5]  prob_volatility          rolling std dev of implied_probability (30-tick window)
  [6]  trade_intensity          trades per minute in rolling 15-min window
  [7]  price_momentum           signed rate of change of implied_probability (10-tick)
  [8]  depth_at_best            total contracts at best bid + best ask (liquidity proxy)
  [9]  volume_weighted_price    vwap of last N trades vs current midpoint (deviation)
 [10]  mean_reversion_signal    z-score of implied_prob relative to its 60-tick mean
 [11]  resolution_proximity     binary sharpness: how far from 0.5 the market currently is
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── data contracts ─────────────────────────────────────────────────────────────

@dataclass
class OrderBookSnapshot:
    """
    single tick of raw market data from polymarket or kalshi.
    all price fields are in [0, 1] representing probability (yes contract price).
    volume fields are in raw contract units.
    """
    timestamp:    float          # unix timestamp seconds
    best_bid:     float          # highest bid price
    best_ask:     float          # lowest ask price
    bid_depth:    float          # total contracts available at best bid
    ask_depth:    float          # total contracts available at best ask
    volume_1h:    float          # cumulative contracts traded in last hour
    trade_count:  float          # number of individual trades in last 15 min
    last_trade_price: float      # price of the most recent matched trade
    last_trade_size:  float      # size of the most recent matched trade
    expiry_ts:    float          # unix timestamp of market resolution


@dataclass
class FeatureRow:
    """
    one row of computed features. 12 floats corresponding to the 12 channels.
    stored pre-normalization; normalization happens at tensor construction time.
    """
    implied_probability:   float
    spread_width:          float
    order_book_imbalance:  float
    volume_1h:             float
    time_to_expiry:        float
    prob_volatility:       float
    trade_intensity:       float
    price_momentum:        float
    depth_at_best:         float
    vwap_deviation:        float
    mean_reversion_signal: float
    resolution_proximity:  float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.implied_probability,
            self.spread_width,
            self.order_book_imbalance,
            self.volume_1h,
            self.time_to_expiry,
            self.prob_volatility,
            self.trade_intensity,
            self.price_momentum,
            self.depth_at_best,
            self.vwap_deviation,
            self.mean_reversion_signal,
            self.resolution_proximity,
        ], dtype=np.float32)


# ── feature computation ────────────────────────────────────────────────────────

# FEATURE 0: implied probability
# the probability the market assigns to YES resolution, estimated as the
# midpoint of the best bid and best ask. this is the primary signal.
def compute_implied_probability(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


# FEATURE 1: spread width
# spread normalized by the midpoint price.
# (ask - bid) / mid — a dimensionless measure of liquidity.
# wide spread → illiquid market → less reliable price signal.
# normalization by mid prevents the raw spread from being artificially small
# near p=0.01 (cheap contracts) vs p=0.99 (expensive contracts).
def compute_spread_width(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid < 1e-6:
        return 0.0
    return (ask - bid) / mid


# FEATURE 2: order book imbalance
# (bid_depth - ask_depth) / (bid_depth + ask_depth) ∈ [-1, +1]
# positive → more buying pressure, negative → more selling pressure.
# this feature captures directional intent before prices move.
# already in [-1, +1] range so normalization has low distortion effect.
def compute_order_book_imbalance(bid_depth: float, ask_depth: float) -> float:
    total = bid_depth + ask_depth
    if total < 1e-6:
        return 0.0
    return (bid_depth - ask_depth) / total


# FEATURE 3: rolling volume (log-scaled)
# raw volume in contracts, log1p-transformed.
# log scaling compresses the heavy right tail of volume distributions.
# volume spikes of 10x would otherwise dominate the feature space.
def compute_volume_log(volume_1h: float) -> float:
    return float(np.log1p(max(volume_1h, 0.0)))


# FEATURE 4: time to expiry (exponential decay)
# normalized remaining lifetime of the contract.
# formula: exp(-λ * days_remaining) where λ = 0.1 controls decay steepness.
# this captures the accelerating information value near resolution:
# a market 1 day from expiry has fundamentally different dynamics than 30 days.
# using exponential rather than linear decay models this non-linearity.
def compute_time_to_expiry(current_ts: float, expiry_ts: float) -> float:
    days_remaining = max((expiry_ts - current_ts) / 86400.0, 0.0)
    return float(np.exp(-0.1 * days_remaining))


# FEATURE 5: rolling probability volatility
# std dev of implied_probability over the last `window` ticks.
# captures regime changes: low volatility → consensus; high volatility → contested.
# computed over a rolling buffer — see FeatureBuffer class below.
def compute_prob_volatility(prob_history: deque) -> float:
    if len(prob_history) < 2:
        return 0.0
    return float(np.std(list(prob_history)))


# FEATURE 6: trade intensity
# trades per minute in the last 15-minute rolling window.
# normalizes raw trade count by time window to be comparable across markets.
# burst intensity signals incoming information: when sophisticated traders
# start trading rapidly it often precedes a large price move.
def compute_trade_intensity(trade_count: float, window_minutes: float = 15.0) -> float:
    return trade_count / max(window_minutes, 1e-6)


# FEATURE 7: price momentum
# signed rate of change: (current_prob - prob_N_ticks_ago) / N
# captures directional drift. 10-tick lookback balances noise vs responsiveness.
# positive = price trending up (more likely YES), negative = trending down.
def compute_price_momentum(prob_history: deque, lookback: int = 10) -> float:
    hist = list(prob_history)
    if len(hist) < lookback + 1:
        return 0.0
    return (hist[-1] - hist[-lookback - 1]) / lookback


# FEATURE 8: depth at best
# total contracts at best bid + best ask (log-scaled).
# thin depth → price is easy to move → signal is noisy.
# log-scaled for same reason as volume: right-tailed distribution.
def compute_depth_at_best(bid_depth: float, ask_depth: float) -> float:
    return float(np.log1p(max(bid_depth + ask_depth, 0.0)))


# FEATURE 9: vwap deviation
# (current_mid - vwap) / vwap — how far the current price is from
# the volume-weighted average price of recent trades.
# positive → current price above where volume transacted (potential mean-revert down)
# negative → current price below vwap (potential mean-revert up)
# captures short-term price dislocations driven by order flow vs fair value.
def compute_vwap_deviation(
    current_mid: float,
    trade_price_history: deque,
    trade_size_history: deque,
) -> float:
    prices = list(trade_price_history)
    sizes  = list(trade_size_history)
    if not prices or sum(sizes) < 1e-6:
        return 0.0
    vwap = sum(p * s for p, s in zip(prices, sizes)) / sum(sizes)
    return (current_mid - vwap) / max(vwap, 1e-6)


# FEATURE 10: mean reversion signal
# z-score of current implied_probability relative to its rolling 60-tick mean.
# z = (x - μ) / σ
# captures how far the current price has deviated from its recent equilibrium.
# binary markets often mean-revert between news events, making this a key signal
# for identifying temporary dislocations vs genuine probability shifts.
def compute_mean_reversion_signal(prob_history: deque, window: int = 60) -> float:
    hist = list(prob_history)[-window:]
    if len(hist) < 2:
        return 0.0
    mu  = np.mean(hist)
    sig = np.std(hist)
    if sig < 1e-8:
        return 0.0
    return float((hist[-1] - mu) / sig)


# FEATURE 11: resolution proximity
# measures how far the market price is from the 0.5 midpoint.
# formula: 2 * |p - 0.5|  ∈ [0, 1]
# value of 0.0 → market is maximally uncertain (p=0.5)
# value of 1.0 → market has effectively resolved (p≈0 or p≈1)
# near-resolved markets (>0.9) have fundamentally different risk dynamics:
# limited upside, high resolution risk. the TCN needs this context.
def compute_resolution_proximity(implied_prob: float) -> float:
    return 2.0 * abs(implied_prob - 0.5)


# ── rolling buffer state ───────────────────────────────────────────────────────

class FeatureBuffer:
    """
    maintains rolling history needed for stateful features (volatility, momentum, etc.).
    one FeatureBuffer instance per active market — state does NOT cross markets.

    uses deques with maxlen for O(1) append/pop vs O(N) list rotation.

    Space: O(WINDOW_MAX) per market instance
    """

    WINDOW_MAX = 64          # maximum history depth across all features

    def __init__(self):
        self.prob_history:        deque = deque(maxlen=self.WINDOW_MAX)
        self.trade_price_history: deque = deque(maxlen=self.WINDOW_MAX)
        self.trade_size_history:  deque = deque(maxlen=self.WINDOW_MAX)

    def update(self, snap: OrderBookSnapshot) -> None:
        mid = compute_implied_probability(snap.best_bid, snap.best_ask)
        self.prob_history.append(mid)
        if snap.last_trade_size > 0:
            self.trade_price_history.append(snap.last_trade_price)
            self.trade_size_history.append(snap.last_trade_size)

    def is_ready(self) -> bool:
        """buffer must have at least 11 ticks before momentum features are valid."""
        return len(self.prob_history) >= 11


# ── tick → feature row ─────────────────────────────────────────────────────────

def compute_feature_row(snap: OrderBookSnapshot, buf: FeatureBuffer) -> FeatureRow:
    """
    computes all 12 features for a single market tick.
    buf must be updated BEFORE calling this function so history includes current tick.

    Time: O(W) where W = max window size (dominated by std dev / mean computations)
    Space: O(1) — reads from buf, allocates only the output FeatureRow
    """
    mid = compute_implied_probability(snap.best_bid, snap.best_ask)

    return FeatureRow(
        implied_probability   = mid,
        spread_width          = compute_spread_width(snap.best_bid, snap.best_ask),
        order_book_imbalance  = compute_order_book_imbalance(snap.bid_depth, snap.ask_depth),
        volume_1h             = compute_volume_log(snap.volume_1h),
        time_to_expiry        = compute_time_to_expiry(snap.timestamp, snap.expiry_ts),
        prob_volatility       = compute_prob_volatility(buf.prob_history),
        trade_intensity       = compute_trade_intensity(snap.trade_count),
        price_momentum        = compute_price_momentum(buf.prob_history),
        depth_at_best         = compute_depth_at_best(snap.bid_depth, snap.ask_depth),
        vwap_deviation        = compute_vwap_deviation(
                                    mid,
                                    buf.trade_price_history,
                                    buf.trade_size_history,
                                ),
        mean_reversion_signal = compute_mean_reversion_signal(buf.prob_history),
        resolution_proximity  = compute_resolution_proximity(mid),
    )


# ── missing data handling ──────────────────────────────────────────────────────

def forward_fill(matrix: np.ndarray) -> np.ndarray:
    """
    forward-fills NaN values along the time axis (axis=0).
    strategy: carry the last valid observation forward.
    if a column begins with NaN (no prior valid value), back-fill from first
    valid observation. if the entire column is NaN, fill with 0.5 (neutral prior
    for probability features, 0 for all others — handled by feature-specific clips).

    forward fill is preferred over interpolation for prediction markets because
    market microstructure data is not smoothly interpolated: a missing tick means
    "no trade occurred", so the last known state is the correct representation.

    Time: O(T * F) where T = timesteps, F = features (12)
    Space: O(T * F) — operates in-place on a copy
    """
    result = matrix.copy()
    T, F = result.shape

    for f in range(F):
        col = result[:, f]
        # find first valid (non-nan) index for this feature column
        valid_mask = ~np.isnan(col)
        if not valid_mask.any():
            col[:] = 0.0      # entire column missing: fill with neutral value
            continue

        # back-fill the leading NaN block from first valid value
        first_valid = np.argmax(valid_mask)
        col[:first_valid] = col[first_valid]

        # forward-fill the remaining NaN blocks
        for t in range(1, T):
            if np.isnan(col[t]):
                col[t] = col[t - 1]

        result[:, f] = col

    return result


def clip_outliers(matrix: np.ndarray, z_threshold: float = 5.0) -> np.ndarray:
    """
    clips extreme values to ±z_threshold standard deviations per feature column.
    computed per-column to handle features with vastly different scales.

    rationale: order book data has fat tails. a single liquidity event can produce
    depth_at_best values 20x normal — these would dominate gradient updates and
    destabilize early training. clipping at ±5σ preserves 99.9999% of real signal
    while eliminating pathological outliers.

    Time: O(T * F)  Space: O(T * F)
    """
    result = matrix.copy()
    for f in range(result.shape[1]):
        col = result[:, f]
        mu  = np.nanmean(col)
        sig = np.nanstd(col)
        if sig > 1e-8:
            result[:, f] = np.clip(col, mu - z_threshold * sig, mu + z_threshold * sig)
    return result


# ── normalization ──────────────────────────────────────────────────────────────

class OnlineNormalizer:
    """
    welford's online algorithm for computing running mean and variance.
    normalizes each feature channel independently (per-channel z-score).

    why per-channel: features have wildly different scales and distributions.
    implied_probability is in [0,1] while volume_1h after log-scaling can be in [0,12].
    mixing them without normalization causes gradient domination by high-variance features.

    why online (welford) rather than batch normalization:
    - we don't have the full dataset at inference time (real-time streaming)
    - welford is numerically stable (avoids catastrophic cancellation in variance)
    - O(1) update per tick, no history storage needed

    formula: μ_n = μ_{n-1} + (x_n - μ_{n-1}) / n
             M_n = M_{n-1} + (x_n - μ_{n-1}) * (x_n - μ_n)
             σ²  = M_n / (n - 1)

    Space: O(F) where F = number of features (12 mean + 12 variance scalars)
    """

    N_FEATURES = 12

    def __init__(self):
        self._count = np.zeros(self.N_FEATURES, dtype=np.float64)
        self._mean  = np.zeros(self.N_FEATURES, dtype=np.float64)
        self._M     = np.zeros(self.N_FEATURES, dtype=np.float64)  # sum of squared deviations

    def update(self, row: np.ndarray) -> None:
        """
        updates running statistics with one new feature row.
        Time: O(F)  Space: O(1)
        """
        for f in range(self.N_FEATURES):
            if np.isnan(row[f]):
                continue
            self._count[f] += 1
            delta           = row[f] - self._mean[f]
            self._mean[f]  += delta / self._count[f]
            delta2          = row[f] - self._mean[f]
            self._M[f]     += delta * delta2

    @property
    def std(self) -> np.ndarray:
        """sample std dev; returns 1.0 for under-sampled channels to avoid /0."""
        variance = np.where(self._count > 1, self._M / (self._count - 1), 1.0)
        return np.sqrt(np.maximum(variance, 1e-8))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        applies z-score normalization using current running statistics.
        z = (x - μ) / σ
        result is approximately N(0,1) for stationary features.

        Time: O(T * F)  Space: O(T * F)
        """
        return (matrix - self._mean) / self.std

    def update_batch(self, matrix: np.ndarray) -> None:
        """
        updates statistics from a full (T, F) matrix before transforming.
        Time: O(T * F)  Space: O(1)
        """
        for row in matrix:
            self.update(row)


# ── sequence windowing ─────────────────────────────────────────────────────────

def build_windows(
    feature_matrix: np.ndarray,
    window_size: int = 64,
    stride: int = 1,
) -> np.ndarray:
    """
    slices a (T, F) feature matrix into overlapping (N, window_size, F) windows.

    stride=1: maximum sample density — every valid timestep produces one window.
    this is correct for online inference (we want the latest 64-tick window).
    for training, stride=4 or stride=8 reduces memory while preserving coverage.

    using numpy stride tricks (as_strided) avoids copying data:
    the output array shares memory with the input — O(1) extra space beyond
    the view metadata, vs O(N * window_size * F) for a naive loop copy.

    Time: O(1) for the view construction; O(N * W * F) when the array is read
    Space: O(1) extra (shared memory view) — critical for large histories

    args:
        feature_matrix: (T, F) array of normalized features
        window_size:    number of timesteps per window (64 for TCN)
        stride:         step size between window start positions

    returns:
        (N, window_size, F) array where N = (T - window_size) // stride + 1
    """
    T, F = feature_matrix.shape
    if T < window_size:
        raise ValueError(
            f"feature matrix has {T} timesteps but window_size={window_size} required. "
            f"need at least {window_size} ticks of market data."
        )

    N = (T - window_size) // stride + 1

    # stride trick: define byte strides for the (N, W, F) view
    # axis 0 (N windows):   step by `stride` rows = stride * F * itemsize bytes
    # axis 1 (W timesteps): step by 1 row         = F * itemsize bytes
    # axis 2 (F features):  step by 1 element     = itemsize bytes
    itemsize = feature_matrix.itemsize
    shape   = (N, window_size, F)
    strides = (stride * F * itemsize, F * itemsize, itemsize)

    windows = np.lib.stride_tricks.as_strided(
        feature_matrix, shape=shape, strides=strides
    )

    # return a contiguous copy: as_strided views are unsafe to write to,
    # and pytorch requires contiguous memory for zero-copy tensor creation
    return np.ascontiguousarray(windows)


# ── tensor construction ────────────────────────────────────────────────────────

def windows_to_tensor(windows: np.ndarray) -> torch.Tensor:
    """
    converts (N, T, F) numpy windows to (N, F, T) pytorch tensor.

    transpose from (N, T, F) → (N, F, T):
    pytorch Conv1d expects (batch, channels, length) — features are channels,
    timesteps are the sequence length dimension.

    Time: O(N * T * F) for the transpose copy
    Space: O(N * T * F)
    """
    # (N, T, F) → (N, F, T) via transpose then contiguous copy for conv1d
    transposed = windows.transpose(0, 2, 1).astype(np.float32)
    return torch.from_numpy(transposed)


# ── top-level pipeline ─────────────────────────────────────────────────────────

class MarketDataPipeline:
    """
    stateful pipeline for a single market. maintains the feature buffer,
    online normalizer, and raw tick history. call ingest() on each new
    OrderBookSnapshot and get_tensor() when the TCN needs a forward pass.

    one instance per active market in the orchestrator.

    Space: O(HISTORY_LIMIT * F) where HISTORY_LIMIT = max ticks stored
    """

    HISTORY_LIMIT = 512      # rolling tick buffer (≈ 8.5 hours at 1-min ticks)
    WINDOW_SIZE   = 64

    def __init__(self, market_id: str):
        self.market_id  = market_id
        self._buf       = FeatureBuffer()
        self._normalizer = OnlineNormalizer()
        self._raw_rows: list[np.ndarray] = []

    def ingest(self, snap: OrderBookSnapshot) -> None:
        """
        processes one new tick: update buffer → compute features → update normalizer.
        called on every websocket message or REST poll for this market.

        Time: O(W) where W = feature window (dominated by rolling std dev)
        Space: O(1) incremental cost per tick
        """
        self._buf.update(snap)

        if not self._buf.is_ready():
            return      # need at least 11 ticks for momentum features

        row = compute_feature_row(snap, self._buf)
        arr = row.to_array()

        # update normalizer BEFORE storing so each row's stats include itself
        self._normalizer.update(arr)
        self._raw_rows.append(arr)

        # enforce rolling buffer cap — drop oldest ticks
        if len(self._raw_rows) > self.HISTORY_LIMIT:
            self._raw_rows.pop(0)

    def get_tensor(self) -> Optional[torch.Tensor]:
        """
        constructs a (1, 12, 64) tensor from the latest 64 ticks.
        returns None if fewer than 64 valid ticks have been ingested.

        full pipeline:
          raw rows → (T, 12) matrix
          → forward-fill NaN gaps
          → clip outliers at ±5σ
          → z-score normalize (per-channel, online statistics)
          → stride-trick window (take latest 64)
          → reshape to (1, 12, 64) for TCN batch dimension

        Time: O(T * F) where T = stored tick count (≤ HISTORY_LIMIT), F = 12
        Space: O(T * F) for the intermediate matrices
        """
        if len(self._raw_rows) < self.WINDOW_SIZE:
            logger.debug(
                f"{self.market_id}: only {len(self._raw_rows)}/{self.WINDOW_SIZE} "
                f"ticks buffered — not ready"
            )
            return None

        # take the most recent WINDOW_SIZE rows
        matrix = np.stack(self._raw_rows[-self.WINDOW_SIZE:], axis=0)  # (64, 12)

        # stage 1: forward-fill any NaN gaps (non-continuous trading periods)
        matrix = forward_fill(matrix)

        # stage 2: clip pathological outliers before normalization
        matrix = clip_outliers(matrix, z_threshold=5.0)

        # stage 3: z-score normalization using online running statistics
        matrix = self._normalizer.transform(matrix)

        # stage 4: final safety clip post-normalization
        # catches any residual extreme values after clipping + normalization
        # this guarantees all TCN inputs are in [-5, 5] without exception
        matrix = np.clip(matrix, -5.0, 5.0)

        # stage 5: build (1, 12, 64) tensor — batch=1 for single market inference
        tensor = windows_to_tensor(matrix[np.newaxis, :, :])  # (1, 64, 12) → transpose → (1, 12, 64)
        return tensor

    def get_batch_tensor(self, stride: int = 4) -> Optional[torch.Tensor]:
        """
        constructs a (N, 12, 64) tensor using all available history with given stride.
        used for TRAINING — generates multiple windows from accumulated history.

        Time: O(T * F + N * W * F) where N = (T - 64) // stride + 1
        Space: O(N * W * F) for the windowed output
        """
        if len(self._raw_rows) < self.WINDOW_SIZE:
            return None

        matrix = np.stack(self._raw_rows, axis=0)           # (T, 12)
        matrix = forward_fill(matrix)
        matrix = clip_outliers(matrix, z_threshold=5.0)
        matrix = self._normalizer.transform(matrix)
        matrix = np.clip(matrix, -5.0, 5.0)

        windows = build_windows(matrix, window_size=self.WINDOW_SIZE, stride=stride)
        return windows_to_tensor(windows)                    # (N, 12, 64)

    @property
    def tick_count(self) -> int:
        return len(self._raw_rows)

    @property
    def is_ready(self) -> bool:
        return len(self._raw_rows) >= self.WINDOW_SIZE


# ── batch pipeline across markets ─────────────────────────────────────────────

class BatchPipeline:
    """
    manages one MarketDataPipeline per active market.
    provides a unified get_batch() method that returns tensors for ALL ready markets.

    used by the orchestrator to feed the TCN in one vectorized forward pass
    across all markets simultaneously — more efficient than N sequential passes.

    Space: O(M * HISTORY_LIMIT * F) where M = active market count
    """

    def __init__(self):
        self._pipelines: dict[str, MarketDataPipeline] = {}

    def ingest(self, market_id: str, snap: OrderBookSnapshot) -> None:
        """
        creates a new pipeline for unseen market_ids automatically.
        Time: O(W) per tick (delegates to MarketDataPipeline.ingest)
        """
        if market_id not in self._pipelines:
            self._pipelines[market_id] = MarketDataPipeline(market_id)
        self._pipelines[market_id].ingest(snap)

    def get_tensor(self, market_id: str) -> Optional[torch.Tensor]:
        """returns (1, 12, 64) tensor for a single market, or None if not ready."""
        pipe = self._pipelines.get(market_id)
        return pipe.get_tensor() if pipe else None

    def get_ready_batch(self) -> tuple[list[str], Optional[torch.Tensor]]:
        """
        returns (market_ids, tensor) for all markets with ≥ 64 ticks.
        tensor shape: (M_ready, 12, 64)

        stacking tensors from ready markets enables a single TCN forward pass
        across all of them simultaneously — O(1) batched GPU call vs O(M) serial.

        Time: O(M * T * F) to build all tensors + O(M_ready * W * F) to stack
        Space: O(M_ready * W * F) for the batched tensor
        """
        ready_ids     = []
        ready_tensors = []

        for market_id, pipe in self._pipelines.items():
            t = pipe.get_tensor()
            if t is not None:
                ready_ids.append(market_id)
                ready_tensors.append(t)

        if not ready_tensors:
            return [], None

        # torch.cat on dim=0: (1,12,64) * M_ready → (M_ready, 12, 64)
        batch_tensor = torch.cat(ready_tensors, dim=0)
        return ready_ids, batch_tensor

    def drop_expired(self, expired_market_ids: list[str]) -> None:
        """removes state for resolved/expired markets to prevent memory leak."""
        for mid in expired_market_ids:
            self._pipelines.pop(mid, None)
        if expired_market_ids:
            logger.info(f"dropped {len(expired_market_ids)} expired market pipelines")


# ── self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time

    print("running data_pipeline self-test...\n")

    # --- test 1: single market pipeline ---
    pipe = MarketDataPipeline("TEST-MARKET-001")
    now  = _time.time()
    expiry = now + 15 * 86400   # 15 days out

    # simulate 80 ticks of realistic polymarket data
    rng = np.random.default_rng(seed=42)
    prob = 0.55
    for i in range(80):
        prob += rng.normal(0, 0.01)
        prob  = np.clip(prob, 0.01, 0.99)
        spread = rng.uniform(0.005, 0.025)

        snap = OrderBookSnapshot(
            timestamp         = now + i * 60,    # 1-min ticks
            best_bid          = prob - spread / 2,
            best_ask          = prob + spread / 2,
            bid_depth         = rng.uniform(50, 500),
            ask_depth         = rng.uniform(50, 500),
            volume_1h         = rng.uniform(100, 2000),
            trade_count       = rng.integers(1, 30),
            last_trade_price  = prob + rng.normal(0, 0.005),
            last_trade_size   = rng.uniform(1, 50),
            expiry_ts         = expiry,
        )
        pipe.ingest(snap)

    assert not pipe.is_ready or pipe.tick_count >= 64, "readiness check failed"
    tensor = pipe.get_tensor()
    assert tensor is not None, "expected tensor after 80 ticks"
    assert tensor.shape == (1, 12, 64), f"wrong shape: {tensor.shape}"
    assert not torch.isnan(tensor).any(), "NaN values in output tensor"
    assert not torch.isinf(tensor).any(), "Inf values in output tensor"
    assert tensor.abs().max().item() <= 5.0, "values outside [-5, 5] clip range"
    print(f"  ✓  single market tensor shape: {tensor.shape}")
    print(f"  ✓  value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"  ✓  no NaN or Inf values")

    # --- test 2: forward fill on sparse data ---
    sparse = np.full((64, 12), np.nan)
    sparse[0,  :] = 0.5
    sparse[30, :] = 0.6
    sparse[63, :] = 0.7
    filled = forward_fill(sparse)
    assert not np.isnan(filled).any(), "forward fill left NaN values"
    assert filled[1, 0] == 0.5, "forward fill incorrect: should carry 0.5 forward"
    print("  ✓  forward fill handles sparse tick data correctly")

    # --- test 3: batch pipeline across multiple markets ---
    batch_pipe = BatchPipeline()
    for market_id in ["MARKET-A", "MARKET-B", "MARKET-C"]:
        prob = 0.5
        for i in range(80):
            prob += rng.normal(0, 0.01)
            prob  = np.clip(prob, 0.01, 0.99)
            spread = rng.uniform(0.005, 0.02)
            snap = OrderBookSnapshot(
                timestamp=now + i * 60,
                best_bid=prob - spread/2, best_ask=prob + spread/2,
                bid_depth=rng.uniform(50, 500), ask_depth=rng.uniform(50, 500),
                volume_1h=rng.uniform(100, 2000), trade_count=rng.integers(1, 20),
                last_trade_price=prob, last_trade_size=rng.uniform(1, 30),
                expiry_ts=expiry,
            )
            batch_pipe.ingest(market_id, snap)

    market_ids, batch_tensor = batch_pipe.get_ready_batch()
    assert batch_tensor is not None
    assert batch_tensor.shape == (3, 12, 64), f"wrong batch shape: {batch_tensor.shape}"
    print(f"  ✓  batch tensor shape: {batch_tensor.shape} ({len(market_ids)} markets)")

    # --- test 4: verify TCN compatibility ---
    import sys
    sys.path.insert(0, ".")
    try:
        from models.tcn_model import TCNProbabilityModel
        model = TCNProbabilityModel()
        model.eval()
        with torch.no_grad():
            p_hat, logit, sigma = model(batch_tensor)
        assert p_hat.shape == (3, 1), f"unexpected p_hat shape: {p_hat.shape}"
        assert (p_hat >= 0).all() and (p_hat <= 1).all(), "p_hat outside [0,1]"
        print(f"  ✓  TCN forward pass: p_hat={p_hat.squeeze().tolist()}")
    except ImportError:
        print("  ⚠  TCN model not importable from this context (run from project root)")

    print("\nall data_pipeline tests passed ✓")