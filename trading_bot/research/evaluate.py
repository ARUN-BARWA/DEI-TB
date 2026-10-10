"""Cost-aware signal evaluation, applied identically to every model (baselines, Kronos, ensembles).

This is a research evaluator, not the event-driven backtester (Phase 8). The protocol is deliberately
conservative and simple enough to audit:

* A model gives a score s_t at each decision time (bar close). Positive = long, negative = short.
* A trade opens when |s_t| >= theta. **theta is chosen on the validation period only**; if no
  threshold is profitable on validation, the model abstains on test (theta = inf).
* Entry at the close of bar t + delay, exit at the close of bar t + delay + h. With a 1-minute price
  lookup and ``lag_minutes`` > 0, both prices are instead the last trade ``lag_minutes`` after those
  times (execution latency; also exposes bid-ask-bounce effects at the bar close). Taker in and out:
  every trade pays the full round-trip cost (fees incl. GST + spread proxy) plus assumed funding
  for its holding time, whatever its direction.
* Trades never overlap within a symbol (the next entry is allowed at the exit bar), and never span a
  data hole (entry and exit must be in the decision bar's segment).
* Results are in log-return units for one unit of notional, without compounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADE_COLUMNS = ["symbol", "entry_time", "exit_time", "side", "gross", "net", "hold_bars"]
QUANTILE_GRID = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)


@dataclass(frozen=True)
class TradeRules:
    h: int
    round_trip_cost: float
    funding_per_bar: float = 0.0  # assumed funding charged per bar held, any side
    delay: int = 0  # bars between decision and entry (latency robustness)
    lag_minutes: int = 0  # extra minutes before entry and exit (needs a 1-minute price lookup)
    extend: bool = False  # at each scheduled exit, keep the position (no new round trip) if the signal still agrees


def minute_price_lookup(minute_bars: pd.DataFrame):
    """Last traded price just before time T: the close of the 1-minute bar that opened at T - 1 minute.

    Returns NaN where that minute is not in the data (holes), so such trades are skipped.
    """
    price = minute_bars["close"].copy()
    price.index = price.index + pd.Timedelta("1min")

    def at(times) -> np.ndarray:
        return price.reindex(pd.DatetimeIndex(times)).to_numpy(dtype=float)

    return at


def simulate_trades(frame: pd.DataFrame, scores: np.ndarray, theta: float, rules: TradeRules,
                    entry_mask: np.ndarray | None = None, max_exit_time: pd.Timestamp | None = None,
                    symbol: str = "", price_at=None) -> pd.DataFrame:
    """Non-overlapping fixed-horizon trades for one symbol. ``frame`` rows are that symbol's bars in time order.

    Needs columns ``close``, ``segment_id``, ``decision_time``. ``entry_mask`` limits which rows
    may open trades (e.g. the test period); exits may use later rows up to ``max_exit_time``.
    """
    close = frame["close"].to_numpy(dtype=float)
    seg = frame["segment_id"].to_numpy()
    dtime = pd.DatetimeIndex(frame["decision_time"])
    n = len(frame)
    lag = pd.Timedelta(minutes=rules.lag_minutes)
    if price_at is not None:
        px = price_at(dtime + lag)
    elif rules.lag_minutes:
        raise ValueError("lag_minutes needs a 1-minute price lookup (price_at)")
    else:
        px = close
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s) & (np.abs(s) >= theta) & (s != 0)
    if entry_mask is not None:
        ok &= np.asarray(entry_mask, dtype=bool)
    rows = []
    next_free = 0
    for i in np.flatnonzero(ok):
        if i < next_free:
            continue
        e, x = i + rules.delay, i + rules.delay + rules.h
        if x >= n:
            break
        if seg[e] != seg[i] or seg[x] != seg[i]:
            continue  # outcome not observable across a data hole
        if max_exit_time is not None and dtime[x] + lag > max_exit_time:
            continue
        if not (np.isfinite(px[e]) and np.isfinite(px[x])):
            continue
        side = 1.0 if s[i] > 0 else -1.0
        if rules.extend:
            # Decision at the scheduled exit bar x uses only information at x: same direction and above
            # threshold -> hold another h bars instead of closing and re-opening (saves a round trip).
            while True:
                nx = x + rules.h
                if not (ok[x] and np.sign(s[x]) == side and nx < n and seg[nx] == seg[i] and np.isfinite(px[nx])):
                    break
                if max_exit_time is not None and dtime[nx] + lag > max_exit_time:
                    break
                x = nx
        gross = side * (np.log(px[x]) - np.log(px[e]))
        net = gross - rules.round_trip_cost - rules.funding_per_bar * (x - e)
        rows.append((symbol, dtime[e] + lag, dtime[x] + lag, side, gross, net, x - e))
        next_free = x
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def t_stat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def choose_threshold(val_frames: dict[str, pd.DataFrame], val_scores: dict[str, np.ndarray],
                     val_masks: dict[str, np.ndarray], rules: dict[str, TradeRules],
                     max_exit_time: pd.Timestamp, quantiles=QUANTILE_GRID, min_trades: int = 20,
                     price_lookups: dict | None = None) -> tuple[float, float]:
    """Pick theta on the validation period: maximise the t-stat of net trade returns, pooled over symbols.

    Returns (theta, validation t-stat). theta = inf (abstain) if nothing has a positive t-stat
    with at least ``min_trades`` trades.
    """
    pooled = np.concatenate([np.abs(val_scores[s][val_masks[s]]) for s in val_frames])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return float("inf"), 0.0
    best = (float("inf"), 0.0)
    for q in quantiles:
        theta = float(np.quantile(pooled, 1 - q))
        if theta == 0:
            theta = float(np.min(pooled[pooled > 0])) if (pooled > 0).any() else float("inf")
        nets = np.concatenate([
            simulate_trades(val_frames[s], val_scores[s], theta, rules[s], val_masks[s], max_exit_time,
                            price_at=(price_lookups or {}).get(s))["net"].to_numpy()
            for s in val_frames
        ])
        t = t_stat(nets)
        if len(nets) >= min_trades and t > best[1]:
            best = (theta, t)
    return best


def sign_flip_pvalue(trades: pd.DataFrame, n_perm: int = 2000, seed: int = 0) -> float:
    """P(mean net with random directions at the same entry times >= observed). Tests direction skill."""
    if trades.empty:
        return float("nan")
    rng = np.random.default_rng(seed)
    raw = (trades["gross"] * trades["side"]).to_numpy()  # underlying long return at each entry
    cost = (trades["gross"] - trades["net"]).to_numpy()
    observed = (trades["net"]).mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(raw)))
    null = (signs * raw - cost).mean(axis=1)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def daily_pnl(trades: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Net P&L booked on exit day, on the given calendar of trading days (zeros included)."""
    if trades.empty:
        return pd.Series(0.0, index=days)
    d = trades.groupby(pd.DatetimeIndex(trades["exit_time"]).floor("D"))["net"].sum()
    return d.reindex(days, fill_value=0.0)


def trade_stats(trades: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    n = len(trades)
    pnl = daily_pnl(trades, days)
    eq = pnl.cumsum()
    dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
    sd = pnl.std(ddof=1)
    out = {
        "trades": n,
        "trades_per_day": round(n / max(len(days), 1), 3),
        "hit_rate": round(float((trades["net"] > 0).mean()), 3) if n else np.nan,
        "mean_gross_bp": round(float(trades["gross"].mean() * 1e4), 2) if n else np.nan,
        "mean_net_bp": round(float(trades["net"].mean() * 1e4), 2) if n else np.nan,
        "t_net": round(t_stat(trades["net"].to_numpy()), 2) if n else np.nan,
        "total_net_pct": round(float(trades["net"].sum() * 100), 2),
        "sharpe_ann": round(float(pnl.mean() / sd * np.sqrt(365)), 2) if n and sd > 0 else np.nan,
        "max_dd_pct": round(dd * 100, 2),
        "long_share": round(float((trades["side"] > 0).mean()), 3) if n else np.nan,
        "p_signflip": round(sign_flip_pvalue(trades), 4) if n else np.nan,
    }
    if n:
        long_ret = (trades["gross"] * trades["side"]).to_numpy()
        cost = (trades["gross"] - trades["net"]).to_numpy()
        out["drift_long_net_bp"] = round(float((long_ret - cost).mean() * 1e4), 2)    # always long at the same times
        out["drift_short_net_bp"] = round(float((-long_ret - cost).mean() * 1e4), 2)  # always short at the same times
    return out


def information_coefficient(scores: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Pearson and Spearman correlation of score with the forward return (finite pairs only)."""
    s, y = np.asarray(scores, float), np.asarray(target, float)
    ok = np.isfinite(s) & np.isfinite(y)
    if ok.sum() < 30 or np.std(s[ok]) == 0:
        return float("nan"), float("nan")
    ic = float(np.corrcoef(s[ok], y[ok])[0, 1])
    ric = float(pd.Series(s[ok]).rank().corr(pd.Series(y[ok]).rank()))
    return ic, ric


def benjamini_hochberg(pvalues) -> np.ndarray:
    """False-discovery-rate q-values (NaN p-values stay NaN and are not counted as tests)."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def ic_t_stat(ic: float, n_rows: int, h: int) -> float:
    """t-stat of an IC from overlapping h-bar labels, using n_rows / h effective observations."""
    n_eff = n_rows / max(h, 1)
    if not np.isfinite(ic) or n_eff <= 3 or abs(ic) >= 1:
        return float("nan")
    return float(ic * np.sqrt((n_eff - 2) / (1 - ic ** 2)))
