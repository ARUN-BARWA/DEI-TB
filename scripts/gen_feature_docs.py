"""Regenerate docs/features.md from the feature registry."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading_bot.features.registry import registry_markdown  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "docs" / "features.md"
    out.write_text(registry_markdown())
    print(f"wrote {out}")
