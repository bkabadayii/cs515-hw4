"""Base dataclasses for project paths and root resolution.

All path construction in the project must derive from ProjectPaths.
No hard-coded absolute paths are allowed in any other module.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field


def _find_project_root() -> pathlib.Path:
    """Walk up from this file until we find pyproject.toml.

    Falls back to the current working directory if no pyproject.toml
    is found, so the package still works when installed in editable
    mode from a different location.
    """
    marker = "pyproject.toml"
    candidate = pathlib.Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / marker).exists():
            return parent
    return pathlib.Path.cwd()


@dataclass(frozen=True)
class ProjectPaths:
    """All canonical project paths derived from the project root.

    Parameters
    ----------
    root:
        Absolute path to the project root directory.  Defaults to the
        directory that contains ``pyproject.toml``.
    """

    root: pathlib.Path = field(default_factory=_find_project_root)

    # ---------- data sub-trees ----------
    @property
    def data_raw(self) -> pathlib.Path:
        """Directory for raw downloaded CSVs."""
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> pathlib.Path:
        """Directory for processed feature files."""
        return self.root / "data" / "processed"

    @property
    def data_splits(self) -> pathlib.Path:
        """Directory for train/val/test NPZ arrays."""
        return self.root / "data" / "splits"

    @property
    def data_metadata(self) -> pathlib.Path:
        """Directory for metadata JSON files."""
        return self.root / "data" / "metadata"

    # ---------- result sub-trees ----------
    @property
    def results_runs(self) -> pathlib.Path:
        """Directory for individual experiment run folders."""
        return self.root / "results" / "runs"

    @property
    def results_aggregate(self) -> pathlib.Path:
        """Directory for aggregated experiment summaries."""
        return self.root / "results" / "aggregate"

    # ---------- figure sub-trees ----------
    @property
    def figures_dataset(self) -> pathlib.Path:
        """Directory for dataset-level figures."""
        return self.root / "figures" / "dataset"

    @property
    def figures_models(self) -> pathlib.Path:
        """Directory for model-level figures."""
        return self.root / "figures" / "models"

    @property
    def figures_final(self) -> pathlib.Path:
        """Directory for final-report figures."""
        return self.root / "figures" / "final_report"

    # ---------- docs ----------
    @property
    def docs(self) -> pathlib.Path:
        """Directory for markdown documentation files."""
        return self.root / "docs"

    def ensure_all(self) -> None:
        """Create all required directories if they do not yet exist."""
        dirs = [
            self.data_raw,
            self.data_processed,
            self.data_splits,
            self.data_metadata,
            self.results_runs,
            self.results_aggregate,
            self.figures_dataset,
            self.figures_models,
            self.figures_final,
            self.docs,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
