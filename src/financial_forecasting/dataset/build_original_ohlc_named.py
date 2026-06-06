"""Phase 3C helper - Re-save original OHLC NPZ files with canonical naming.

The original Phase 2 pipeline saved files as exact_{split}.npz.
Phase 3 uses a naming convention that includes the feature-set name:
exact_original_ohlc_{split}.npz.

This script copies the existing NPZ files to the new canonical names
so that all training scripts can use a consistent file-naming scheme.

The existing files are NOT deleted or modified.
"""

from __future__ import annotations

import shutil

from financial_forecasting.config.base import ProjectPaths


def main() -> None:
    """Copy original OHLC NPZ files to canonical feature-set names."""
    print("=" * 60)
    print("Phase 3C - Re-naming original OHLC NPZ files")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()

    splits = ["train", "val", "test"]

    for split in splits:
        src = paths.data_splits / f"exact_{split}.npz"
        dst = paths.data_splits / f"exact_original_ohlc_{split}.npz"

        if not src.exists():
            raise FileNotFoundError(
                f"Source NPZ not found: {src}\n"
                "Please run build_exact_return_windows.py first."
            )

        if dst.exists():
            print(f"Already exists, skipping: {dst.name}")
        else:
            shutil.copy2(src, dst)
            print(f"Copied {src.name} -> {dst.name}")

    print("\nOriginal OHLC NPZ files available under canonical names.")
    print("=" * 60)


if __name__ == "__main__":
    main()
