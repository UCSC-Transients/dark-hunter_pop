"""Shared rendering primitives for stage diagnostics and paper-ready product figures.

ARCHITECTURE.md §4. Matplotlib is an optional dependency (``pip install -e ".[plot]"``);
callers must tolerate missing matplotlib when figures are disabled or unavailable.
Display-only: excluded from stage ``source_hash`` dependency lists.

Style defaults live in ``config.plotting`` / ``PlottingStyleConfig`` (see ``docs/PLOTS.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_schema import PlottingStyleConfig


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


def default_plotting_style() -> PlottingStyleConfig:
    """Return schema defaults (same values as ``config/fragments/plotting.yaml``)."""
    return PlottingStyleConfig()


def resolve_plotting_style(style: PlottingStyleConfig | None) -> PlottingStyleConfig:
    """Use the provided style or schema defaults."""
    return style if style is not None else default_plotting_style()


def series_style(
    index: int,
    style: PlottingStyleConfig | None = None,
) -> dict[str, Any]:
    """Color / linestyle / marker for series ``index`` (cycles Okabe–Ito + linestyles)."""
    cfg = resolve_plotting_style(style)
    colors = list(cfg.color_cycle) or ["#000000"]
    linestyles = list(cfg.linestyle_cycle) or ["-"]
    markers = list(cfg.marker_cycle) or ["o"]
    return {
        "color": colors[index % len(colors)],
        "linestyle": linestyles[index % len(linestyles)],
        "marker": markers[index % len(markers)],
        "linewidth": float(cfg.line_width),
        "markersize": float(cfg.marker_size),
    }


def apply_axes_style(
    axis: Any,
    style: PlottingStyleConfig | None = None,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    enable_minor_ticks: bool = True,
) -> None:
    """Apply project tick/font/spine defaults to one axes (``docs/PLOTS.md``).

    Inward ticks on all four sides, minor ticks when the projection supports them,
    serif labels at config font sizes. Prefer calling this after plotting data.
    """
    cfg = resolve_plotting_style(style)
    if xlabel is not None:
        axis.set_xlabel(
            xlabel, fontfamily=cfg.font_family, fontsize=cfg.axes_label_fontsize
        )
    if ylabel is not None:
        axis.set_ylabel(
            ylabel, fontfamily=cfg.font_family, fontsize=cfg.axes_label_fontsize
        )
    if title is not None:
        axis.set_title(
            title, fontfamily=cfg.font_family, fontsize=cfg.title_fontsize
        )

    axis.tick_params(
        axis="both",
        which="major",
        right=True,
        top=True,
        width=cfg.tick_width,
        length=cfg.tick_major_length,
        direction=cfg.tick_direction,
        labelsize=cfg.tick_label_fontsize,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        right=True,
        top=True,
        width=cfg.tick_width,
        length=cfg.tick_minor_length,
        direction=cfg.tick_direction,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(cfg.spines_width)

    if enable_minor_ticks:
        try:
            from matplotlib.ticker import AutoMinorLocator

            if axis.get_xaxis().get_scale() == "linear":
                axis.xaxis.set_minor_locator(AutoMinorLocator())
            if axis.get_yaxis().get_scale() == "linear":
                axis.yaxis.set_minor_locator(AutoMinorLocator())
        except (AttributeError, ValueError, TypeError):
            # Geographic / 3D / custom projections may reject minor locators.
            pass

    for label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        label.set_fontfamily(cfg.font_family)


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
    ``values`` must already be finite; callers should drop NaN/inf first.
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
    color: str | None = None,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Write a one-dimensional histogram PNG. Returns None when values are empty.

    Non-finite entries are dropped so ``bins="auto"`` does not raise on NaN ranges
    (e.g. missing eccentricities in NSS orbital blocks).
    """
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    cfg = resolve_plotting_style(style)
    resolved = resolve_histogram_bins(finite, bins, max_bins=max_bins)
    face = color if color is not None else cfg.hist_face_color
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))
    axis.hist(
        finite,
        bins=resolved,
        color=face,
        edgecolor=cfg.hist_edge_color,
        linewidth=cfg.spines_width * 0.5,
    )
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
    return save_figure(fig, path, dpi=dpi)


def plot_sky_mollweide(
    ra_deg: NDArray[np.floating] | Sequence[float] | None,
    dec_deg: NDArray[np.floating] | Sequence[float] | None,
    path: Path,
    *,
    title: str = "sky coverage",
    dpi: int,
    point_size: float | None = None,
    alpha: float | None = None,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Write an equatorial Mollweide sky map. Returns None when coordinates are empty.

    Marker size / alpha are usually supplied from ``diagnostics.sky_map_*``; when
    omitted, a modest default suitable for small samples is used.
    """
    if ra_deg is None or dec_deg is None:
        return None
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    if ra.size == 0 or dec.size == 0 or ra.size != dec.size:
        return None
    finite = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(finite):
        return None
    ra = ra[finite]
    dec = dec[finite]

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    cfg = resolve_plotting_style(style)
    plt = require_pyplot()
    fig = plt.figure(figsize=tuple(cfg.figsize_wide))
    ax = fig.add_subplot(111, projection="mollweide")
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    color = cfg.color_cycle[5] if len(cfg.color_cycle) > 5 else cfg.color_cycle[0]
    ax.scatter(
        coord.ra.wrap_at(180 * u.deg).radian,
        coord.dec.radian,
        s=float(0.1 if point_size is None else point_size),
        alpha=float(0.25 if alpha is None else alpha),
        c=color,
        rasterized=True,
        linewidths=0.0,
    )
    apply_axes_style(
        ax,
        cfg,
        title=title,
        enable_minor_ticks=False,
    )
    return save_figure(fig, path, dpi=dpi)


def plot_overlay_histograms(
    series: Mapping[str, NDArray[np.floating] | Sequence[float]],
    path: Path,
    *,
    xlabel: str,
    title: str,
    dpi: int,
    bins: int | str = "auto",
    max_bins: int | None = None,
    ylabel: str = "density",
    density: bool = True,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Overlay named histograms (El-Badry-style mock vs real panel building block)."""
    cfg = resolve_plotting_style(style)
    prepared: list[tuple[str, NDArray[np.float64]]] = []
    for label, values in series.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            prepared.append((label, finite))
    if not prepared:
        return None

    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))
    for index, (label, arr) in enumerate(prepared):
        sty = series_style(index, cfg)
        resolved = resolve_histogram_bins(arr, bins, max_bins=max_bins)
        axis.hist(
            arr,
            bins=resolved,
            density=density,
            histtype="step",
            linewidth=sty["linewidth"],
            label=label,
            color=sty["color"],
            linestyle=sty["linestyle"],
        )
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
    axis.legend(loc="best", fontsize=cfg.legend_fontsize, prop={"family": cfg.font_family})
    return save_figure(fig, path, dpi=dpi)


def plot_six_panel_grid(
    panels: Mapping[str, Mapping[str, NDArray[np.floating] | Sequence[float]]],
    path: Path,
    *,
    panel_order: Sequence[str],
    dpi: int,
    title: str = "El-Badry-style six-panel comparison",
    bins: int | str = "auto",
    max_bins: int | None = None,
    density: bool = True,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Write a 2×3 grid of overlay histograms (selection-function validation style).

    ``panels`` maps panel name → {series_label → values}. Missing or empty series are
    skipped within a panel; a panel with no data is left blank with an annotation.
    """
    if not panel_order:
        return None
    cfg = resolve_plotting_style(style)
    plt = require_pyplot()
    n = len(panel_order)
    nrows = 2 if n > 3 else 1
    ncols = min(3, n) if n else 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(cfg.figsize_landscape[0] / 7.0 * 4 * ncols, 3.5 * nrows),
    )
    flat = np.atleast_1d(axes).ravel()
    any_drawn = False
    for index, panel_name in enumerate(panel_order):
        axis = flat[index]
        series = panels.get(panel_name, {})
        drawn = False
        series_index = 0
        for label, values in series.items():
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            sty = series_style(series_index, cfg)
            resolved = resolve_histogram_bins(finite, bins, max_bins=max_bins)
            axis.hist(
                finite,
                bins=resolved,
                density=density,
                histtype="step",
                linewidth=sty["linewidth"],
                label=label,
                color=sty["color"],
                linestyle=sty["linestyle"],
            )
            series_index += 1
            drawn = True
            any_drawn = True
        apply_axes_style(axis, cfg, xlabel=panel_name, title=panel_name)
        if drawn:
            axis.legend(
                loc="best",
                fontsize=max(8.0, float(cfg.legend_fontsize) - 4.0),
                prop={"family": cfg.font_family},
            )
        else:
            axis.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontfamily=cfg.font_family,
                fontsize=cfg.tick_label_fontsize,
            )
    for index in range(len(panel_order), len(flat)):
        flat[index].axis("off")
    fig.suptitle(title, fontfamily=cfg.font_family, fontsize=cfg.title_fontsize)
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
    color: str | None = None,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Bar chart for funnel steps, fit-tier coverage, or gate pass rates."""
    if len(labels) != len(values) or not labels:
        return None
    cfg = resolve_plotting_style(style)
    face = color if color is not None else cfg.hist_face_color
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))
    positions = np.arange(len(labels))
    axis.bar(
        positions,
        [float(v) for v in values],
        color=face,
        edgecolor=cfg.hist_edge_color,
        linewidth=cfg.spines_width * 0.5,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        list(labels),
        rotation=30,
        ha="right",
        fontfamily=cfg.font_family,
        fontsize=cfg.tick_label_fontsize,
    )
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
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
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Grouped bar chart (e.g. mock vs real solution-type fractions)."""
    if not labels or not series:
        return None
    n_series = len(series)
    for values in series.values():
        if len(values) != len(labels):
            return None
    cfg = resolve_plotting_style(style)
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.8 / n_series
    for index, (name, values) in enumerate(series.items()):
        sty = series_style(index, cfg)
        offset = (index - 0.5 * (n_series - 1)) * width
        axis.bar(
            positions + offset,
            [float(v) for v in values],
            width=width,
            label=name,
            color=sty["color"],
            edgecolor=cfg.hist_edge_color,
            linewidth=cfg.spines_width * 0.5,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        list(labels),
        rotation=30,
        ha="right",
        fontfamily=cfg.font_family,
        fontsize=cfg.tick_label_fontsize,
    )
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
    axis.legend(loc="best", fontsize=cfg.legend_fontsize, prop={"family": cfg.font_family})
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
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Line plot with optional horizontal threshold (MC/Poisson convergence)."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size == 0 or yy.size == 0 or xx.size != yy.size:
        return None
    cfg = resolve_plotting_style(style)
    sty = series_style(0, cfg)
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))
    axis.plot(
        xx,
        yy,
        marker=sty["marker"],
        linewidth=sty["linewidth"],
        markersize=sty["markersize"],
        color=sty["color"],
        linestyle=sty["linestyle"],
    )
    if threshold is not None:
        axis.axhline(
            float(threshold),
            color=cfg.threshold_color,
            linestyle=cfg.threshold_linestyle,
            linewidth=cfg.line_width,
            label=threshold_label,
        )
        axis.legend(
            loc="best",
            fontsize=cfg.legend_fontsize,
            prop={"family": cfg.font_family},
        )
    if log_x:
        axis.set_xscale("log")
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
    return save_figure(fig, path, dpi=dpi)
