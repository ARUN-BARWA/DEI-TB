"""How good is Kronos as a *forecaster* (separately from whether it helps trading)? Development period only.

    python scripts/kronos_calibration.py --kronos-tag Kronos-mini_paths --timeframes 4h 1h

Per horizon h (bars), using the sampled-path statistics against the realised log return y_ret_h:
    cover_90 / cover_50   share of outcomes inside [q05, q95] / [q25, q75]  (ideal 0.90 / 0.50)
    disp_ratio            std(realised) / mean(predicted std)  (>1: Kronos is over-confident)
    ic_mean, rank_ic      correlation of the path-mean forecast with the outcome
    ic_p_up               correlation of P(up) with the outcome
    hit                   direction accuracy of the path mean (outcomes != 0)
    brier, brier_clim     Brier score of P(up) vs always predicting the training base rate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading_bot.config import load_settings  # noqa: E402
from trading_bot.data.storage import Store  # noqa: E402
from trading_bot.research.evaluate import information_coefficient  # noqa: E402
from trading_bot.research.splits import development_mask  # noqa: E402


def calibration_rows(df: pd.DataFrame, horizons, holdout_start) -> list[dict]:
    rows = []
    for h in horizons:
        if f"kr_r{h}_mean" not in df or f"y_ret_{h}" not in df:
            continue
        m = development_mask(df[f"label_end_{h}"], holdout_start) & df[f"kr_r{h}_mean"].notna().to_numpy() & df[f"y_ret_{h}"].notna().to_numpy()
        d = df.loc[m]
        y = d[f"y_ret_{h}"].to_numpy()
        if len(y) < 50:
            continue
        up = (y > 0).astype(float)
        p_up = d[f"kr_r{h}_p_up"].to_numpy()
        mean = d[f"kr_r{h}_mean"].to_numpy()
        nz = y != 0
        ic, ric = information_coefficient(mean, y)
        ic_p, _ = information_coefficient(p_up, y)
        rows.append({
            "h": h, "n": len(y),
            "cover_90": round(float(((y >= d[f"kr_r{h}_q05"]) & (y <= d[f"kr_r{h}_q95"])).mean()), 3),
            "cover_50": round(float(((y >= d[f"kr_r{h}_q25"]) & (y <= d[f"kr_r{h}_q75"])).mean()), 3),
            "disp_ratio": round(float(np.std(y) / np.mean(d[f"kr_r{h}_std"])), 2),
            "ic_mean": round(ic, 4), "rank_ic": round(ric, 4), "ic_p_up": round(ic_p, 4),
            "hit": round(float((np.sign(mean[nz]) == np.sign(y[nz])).mean()), 3),
            "brier": round(float(np.mean((p_up - up) ** 2)), 4),
            "brier_clim": round(float(np.mean((up.mean() - up) ** 2)), 4),
        })
    return rows


def main() -> None:
    settings = load_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(settings.data_root))
    p.add_argument("--kronos-tag", required=True)
    p.add_argument("--symbols", nargs="+", default=list(settings.combined.symbols) or list(settings.symbols))
    p.add_argument("--timeframes", nargs="+", default=["4h", "1h"])
    args = p.parse_args()
    store = Store(args.root)
    out = []
    for tf in args.timeframes:
        frames = []
        for sym in args.symbols:
            ds = pd.read_parquet(store.dataset_path(sym, tf)).join(pd.read_parquet(store.kronos_path(args.kronos_tag, tf, sym)), how="left")
            horizons = sorted(int(c[4:-5]) for c in ds.columns if c.startswith("kr_r") and c.endswith("_mean"))
            for r in calibration_rows(ds, horizons, settings.research.holdout_start):
                out.append({"tf": tf, "symbol": sym, **r})
            frames.append(ds)
        for r in calibration_rows(pd.concat(frames), horizons, settings.research.holdout_start):
            out.append({"tf": tf, "symbol": "ALL", **r})
    table = pd.DataFrame(out)
    if table.empty:
        sys.exit("no path statistics found; compute features with --paths")
    store.reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(store.reports_dir / f"kronos_calibration_{args.kronos_tag}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
