"""Shared rendering primitives for stage diagnostics and paper-ready product figures.

ARCHITECTURE.md §4. Matplotlib is an optional dependency (``pip install -e ".[plot]"``);
callers must tolerate missing matplotlib when figures are disabled or unavailable.
Display-only: excluded from stage ``source_hash`` dependency lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class MatplotlibUnavailableError(ImportError):
    """Raised when a figure primitive is invoked without matplotlib installed."""


def matplotlib_available() -> bool:
    """Return True when ``matplotlib.pyplot`` can be imported."""
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        return False
    return True


def require_pyplot() -> Any:
    """Import ``matplotlib.pyplot`` or raise ``MatplotlibUnavailableError``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MatplotlibUnavailableError(
            "matplotlib is required for figure output; install with "
            'pip install -e ".[plot]"'
        ) from exc
    return plt


def save_figure(fig: Any, path: Path, *, dpi: int) -> Path:
    """Save a matplotlib figure to ``path`` (parents created) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt = require_pyplot()
    plt.close(fig)
    return path


def resolve_histogram_bins(
    values: NDArray[np.floating],
    bins: int | str = "auto",
    *,
    max_bins: int | None = None,
) -> int | str | NDArray[np.floating]:
    """Resolve histogram binning, capping ``auto``/int counts when ``max_bins`` is set.

    Heavy-tailed large-N samples make Freedman–Diaconis ``bins="auto"`` produce
    hundreds–thousands of sub-pixel bars that render as an empty plot (#96).
    """
    if max_bins is None:
        return bins
    if max_bins < 1:
        raise ValueError(f"max_bins must be >= 1, got {max_bins}")
    if isinstance(bins, int):
        return min(int(bins), int(max_bins))
    if bins == "auto":
        _counts, edges = np.histogram(values, bins="auto")
        n_bins = int(len(edges) - 1)
        if n_bins > int(max_bins):
            return int(max_bins)
        return "auto"
    return bins


def plot_histogram(
    values: NDArray[np.floating] | Sequence[float] | None,
    path: Path,
    *,
    xlabel: str,
    ylabel: str = "count",
    title: str,
    dpi: int,
    bins: int | str = "auto",
    max_bins: int | None = None,
    color: str = "steelblue",
) -> Path | None:
    """Write a one-dimensional histogram PNG. Returns None when values are empty."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    resolved = resolve_histogram_bins(arr, bins, max_bins=max_bins)
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(arr, bins=resolved, color=color, edgecolor="white")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    return save_figure(fig, path, dpi=dpi)


def plot_sky_mollweide(
    ra_deg: NDArray[np.floating] | Sequence[float] | None,
    dec_deg: NDArray[np.floating] | Sequence[float] | None,
    path: Path,
    *,
    title: str = "sky coverage",
    dpi: int,
    point_size: float = 0.1,
    alpha: float = 0.25,
) -> Path | None:
    """Write an equatorial Mollweide sky map. Returns None when coordinates are empty.

    Default ``point_size`` / ``alpha`` match ``diagnostics.sky_map_*`` config and are
    sized for NSS-scale catalogs (#97); override for small demo samples if needed.
    """
    if ra_deg is None or dec_deg is None:
        return None
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    if ra.size == 0 or dec.size == 0 or ra.size != dec.size:
        return None

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    plt = require_pyplot()
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection="mollweide")
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    ax.scatter(
        coord.ra.wrap_at(180 * u.deg).radian,
        coord.dec.radian,
        s=float(point_size),
        alpha=float(alpha),
        rasterized=True,
        linewidths=0.0,
    )
    ax.set_title(title)
    return save_figure(fig, path, dpi=dpi)


def plot_overlay_histograms(
    series: Mapping[str, NDArray[np.floating] | Sequence[float]],
    path: Path,
    *,
    xlabel: str,
    title: str,
    dpi: int,
    bins: int | str = "auto",
    ylabel: str = "density",
    density: bool = True,
) -> Path | None:
    """Overlay named histograms (El-Badry-style mock vs real panel building block)."""
    prepared: list[tuple[str, NDArray[np.float64]]] = []
    for label, values in series.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size:
            prepared.append((label, arr))
    if not prepared:
        return None

    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=(6, 4))
    for label, arr in prepared:
        axis.hist(arr, bins=bins, density=density, histtype="step", linewidth=1.5, label=label)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(loc="best")
    return save_figure(fig, path, dpi=dpi)


def plot_six_panel_grid(
    panels: Mapping[str, Mapping[str, NDArray[np.floating] | Sequence[float]]],
    path: Path,
    *,
    panel_order: Sequence[str],
    dpi: int,
    title: str = "El-Badry-style six-panel comparison",
    bins: int | str = "auto",
    density: bool = True,
) -> Path | None:
    """Write a 2×3 grid of overlay histograms (selection-function validation style).

    ``panels`` maps panel name → {series_label → values}. Missing or empty series are
    skipped within a panel; a panel with no data is left blank with an annotation.
    """
    if not panel_order:
        return None
    plt = require_pyplot()
    n = len(panel_order)
    nrows = 2 if n > 3 else 1
    ncols = min(3, n) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    flat = np.atleast_1d(axes).ravel()
    any_drawn = False
    for index, panel_name in enumerate(panel_order):
        axis = flat[index]
        series = panels.get(panel_name, {})
        drawn = False
        for label, values in series.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            axis.hist(
                arr,
                bins=bins,
                density=density,
                histtype="step",
                linewidth=1.5,
                label=label,
            )
            drawn = True
            any_drawn = True
        axis.set_title(panel_name)
        axis.set_xlabel(panel_name)
        if drawn:
            axis.legend(loc="best", fontsize="small")
        else:
            axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)
    for index in range(len(panel_order), len(flat)):
        flat[index].axis("off")
    fig.suptitle(title)
    if not any_drawn:
        plt.close(fig)
        return None
    return save_figure(fig, path, dpi=dpi)


def plot_categorical_bars(
    labels: Sequence[str],
    values: Sequence[float],
    path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    dpi: int,
    color: str = "steelblue",
) -> Path | None:
    """Bar chart for funnel steps, fit-tier coverage, or gate pass rates."""
    if len(labels) != len(values) or not labels:
        return None
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(labels))
    axis.bar(positions, [float(v) for v in values], color=color, edgecolor="white")
    axis.set_xticks(positions)
    axis.set_xticklabels(list(labels), rotation=30, ha="right")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    return save_figure(fig, path, dpi=dpi)


def plot_grouped_bars(
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    dpi: int,
) -> Path | None:
    """Grouped bar chart (e.g. mock vs real solution-type fractions)."""
    if not labels or not series:
        return None
    n_series = len(series)
    for values in series.values():
        if len(values) != len(labels):
            return None
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.8 / n_series
    for index, (name, values) in enumerate(series.items()):
        offset = (index - 0.5 * (n_series - 1)) * width
        axis.bar(
            positions + offset,
            [float(v) for v in values],
            width=width,
            label=name,
            edgecolor="white",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(list(labels), rotation=30, ha="right")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(loc="best")
    return save_figure(fig, path, dpi=dpi)


def plot_line_with_threshold(
    x: Sequence[float] | NDArray[np.floating],
    y: Sequence[float] | NDArray[np.floating],
    path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    dpi: int,
    threshold: float | None = None,
    threshold_label: str = "threshold",
    log_x: bool = False,
) -> Path | None:
    """Line plot with optional horizontal threshold (MC/Poisson convergence)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size == 0 or yy.size == 0 or xx.size != yy.size:
        return None
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(xx, yy, marker="o", linewidth=1.5)
    if threshold is not None:
        axis.axhline(
            float(threshold),
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=threshold_label,
        )
        axis.legend(loc="best")
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    return save_figure(fig, path, dpi=dpi)
