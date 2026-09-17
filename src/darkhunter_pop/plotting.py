"""Shared rendering primitives for stage diagnostics and paper-ready product figures.

ARCHITECTURE.md §4. Matplotlib is an optional dependency (``pip install -e ".[plot]"``);
callers must tolerate missing matplotlib when figures are disabled or unavailable.
Display-only: excluded from stage ``source_hash`` dependency lists.

Style defaults live in ``config.plotting`` / ``PlottingStyleConfig`` (see ``docs/PLOTS.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping, Sequence

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
    xlim: tuple[float, float] | None = None,
    log_y: bool = False,
) -> Path | None:
    """Write a one-dimensional histogram PNG. Returns None when values are empty.

    Non-finite entries are dropped so ``bins="auto"`` does not raise on NaN ranges
    (e.g. missing eccentricities in NSS orbital blocks). When ``xlim`` is set, only
    values inside ``[xlim[0], xlim[1]]`` are binned and the axis is fixed; counts
    outside the range are appended to the title so heavy tails are not hidden.
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
    face = color if color is not None else cfg.hist_face_color
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))

    display_title = title
    if xlim is not None:
        xmin, xmax = xlim
        if xmin >= xmax:
            raise ValueError(f"xlim must satisfy xmin < xmax, got {xlim}")
        n_above = int(np.sum(finite > xmax))
        n_below = int(np.sum(finite < xmin))
        in_range = finite[(finite >= xmin) & (finite <= xmax)]
        if in_range.size == 0:
            plt.close(fig)
            return None
        n_bins = int(max_bins) if max_bins is not None else 40
        bin_edges = np.linspace(xmin, xmax, n_bins + 1)
        axis.hist(
            in_range,
            bins=bin_edges,
            color=face,
            edgecolor=cfg.hist_edge_color,
            linewidth=cfg.spines_width * 0.5,
        )
        axis.set_xlim(xmin, xmax)
        omitted: list[str] = []
        if n_above:
            omitted.append(f"{n_above} > {xmax:g} M$_\\odot$")
        if n_below:
            omitted.append(f"{n_below} < {xmin:g} M$_\\odot$")
        if omitted:
            display_title = f"{title} ({'; '.join(omitted)} omitted)"
    else:
        resolved = resolve_histogram_bins(finite, bins, max_bins=max_bins)
        axis.hist(
            finite,
            bins=resolved,
            color=face,
            edgecolor=cfg.hist_edge_color,
            linewidth=cfg.spines_width * 0.5,
        )

    if log_y:
        axis.set_yscale("log")
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=display_title)
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
    panel_xlabels: Mapping[str, str] | None = None,
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
        apply_axes_style(
            axis,
            cfg,
            xlabel=(panel_xlabels or {}).get(panel_name, panel_name),
            title=panel_name,
        )
        if drawn:
            axis.legend(
                loc="best",
                fontsize=cfg.legend_fontsize,
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


def measure_text_width_inches(
    text: str, *, font_family: str, fontsize: float
) -> float:
    """Rendered width of ``text`` in inches, for layout that must not overflow.

    Character-count heuristics overflow badly for serif text at label sizes, so
    layout code that has to fit text inside a known width measures instead. The
    measurement uses a throwaway figure and its renderer, then discards both.

    Parameters
    ----------
    text:
        String to measure. Newlines are not handled — measure one line at a time.
    font_family / fontsize:
        Font as it will actually be drawn (``plotting.font_family`` and the
        relevant ``*_fontsize``).

    Returns
    -------
    float
        Width in inches. ``0.0`` for empty text.

    Limitations
    -----------
    The measurement is taken at the probe figure's DPI and assumes the real
    figure renders the same font at the same size — true for the Agg/PDF paths
    used here. Mathtext is measured as mathtext, so a ``$...$`` fragment is
    sized correctly. Falls back to a conservative per-character estimate if the
    backend cannot supply a renderer.
    """
    if not text:
        return 0.0
    plt = require_pyplot()
    probe = plt.figure(figsize=(1.0, 1.0))
    try:
        artist = probe.text(0.0, 0.0, text, fontfamily=font_family, fontsize=fontsize)
        try:
            renderer = probe.canvas.get_renderer()  # type: ignore[attr-defined]
            extent = artist.get_window_extent(renderer=renderer)
        except (AttributeError, RuntimeError, ValueError):
            # Conservative fallback: 0.6 em per character.
            return 0.6 * float(fontsize) / 72.0 * len(text)
        return float(extent.width) / float(probe.dpi)
    finally:
        plt.close(probe)


def wrap_text_to_inches(
    text: str,
    *,
    max_width_inches: float,
    font_family: str,
    fontsize: float,
) -> list[str]:
    """Wrap ``text`` so no rendered line exceeds ``max_width_inches``.

    Blank input lines are preserved as blank output lines, so paragraph and list
    structure in a caption survives wrapping.

    Parameters
    ----------
    text:
        Possibly multi-line text. Each input line is wrapped independently.
    max_width_inches:
        Hard width budget per line, in inches.
    font_family / fontsize:
        Font as it will be drawn.

    Returns
    -------
    list[str]
        Wrapped lines, ready to join with newlines.

    Limitations
    -----------
    Breaks on whitespace only: a single word (or one long unbroken path or hash)
    wider than the budget is emitted on its own over-wide line rather than being
    hyphenated or truncated. Measuring every candidate line makes this O(words)
    in renderer calls, so it is meant for captions, not for bulk labelling.
    """
    if max_width_inches <= 0.0:
        return text.splitlines()
    char_width = measure_text_width_inches(
        "n", font_family=font_family, fontsize=fontsize
    )
    # First-pass wrap by an estimated character budget, then repair any line that
    # still measures too wide. Two passes keep renderer calls bounded.
    est_chars = max(20, int(max_width_inches / max(char_width, 1e-6)))
    import textwrap

    out: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            out.append("")
            continue
        indent = " " * (len(paragraph) - len(paragraph.lstrip(" ")))
        for candidate in textwrap.wrap(
            paragraph,
            width=est_chars,
            subsequent_indent=indent + "  ",
        ) or [""]:
            while (
                measure_text_width_inches(
                    candidate, font_family=font_family, fontsize=fontsize
                )
                > max_width_inches
                and " " in candidate.strip()
            ):
                head, _, tail = candidate.rpartition(" ")
                out.append(head)
                candidate = indent + "  " + tail
            out.append(candidate)
    return out


#: Headroom above the largest sample on a clipped log axis, in decades, so the
#: top curve's markers are not clipped by the frame.
_LOG_Y_HEADROOM_DECADES: Final[float] = 0.5


def dndm_log_ylim(largest: float, decades: float) -> tuple[float, float]:
    """Log y limits keeping ``decades`` of range below ``largest``.

    A soft ``M_TOV`` truncation decays smoothly toward zero, so the raw data
    range on a log axis can exceed a hundred decades and flatten every curve
    against the top of the frame. Clipping to a fixed window keeps the relevant
    dynamic range filling the panel (``docs/PLOTS.md``).

    Parameters
    ----------
    largest:
        Largest positive sample across every drawn curve.
    decades:
        Decades of range to keep below ``largest`` (``plotting.dndm_y_decades``).

    Returns
    -------
    tuple[float, float]
        ``(floor, ceiling)``, with half a decade of headroom above ``largest``.

    Raises
    ------
    ValueError
        If ``largest`` or ``decades`` is not positive — neither is meaningful on
        a log axis.

    Limitations
    -----------
    Curves lying entirely below the floor vanish without annotation. The caller
    is expected to report which classes had no drawable curve.
    """
    if not largest > 0.0:
        raise ValueError(f"largest must be positive on a log axis, got {largest!r}")
    if not decades > 0.0:
        raise ValueError(f"decades must be positive, got {decades!r}")
    return (
        float(largest) * 10.0 ** (-float(decades)),
        float(largest) * 10.0**_LOG_Y_HEADROOM_DECADES,
    )


def plot_dndm_by_class(
    mass_grid_msun: Sequence[float] | NDArray[np.floating],
    total_dndm: Sequence[float] | NDArray[np.floating],
    class_dndm: Mapping[str, Sequence[float] | NDArray[np.floating]],
    path: Path,
    *,
    dpi: int,
    class_order: Sequence[str],
    total_label: str = "total (raw CO)",
    xlabel: str = r"companion mass (M$_{\odot}$)",
    ylabel: str = r"${\rm d}N/{\rm d}M$ (M$_{\odot}^{-1}$)",
    title: str | None = None,
    caption: str | None = None,
    log_x: bool = True,
    log_y: bool = True,
    vlines: Mapping[str, float] | None = None,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Product figure: total ``dN/dM`` with every population class overplotted.

    The deliverable shape of the eventual science result (ARCHITECTURE.md §4
    ``diagnostics``): one panel, the tier-1 raw compact-object total plus each
    tier-2 species-classified curve, discriminated by color **and** linestyle
    **and** marker so the panel survives greyscale printing and color-vision
    deficiency (``docs/PLOTS.md``).

    Parameters
    ----------
    mass_grid_msun:
        Companion-mass grid in solar masses, shared by every curve.
    total_dndm:
        Total (classification-independent) ``dN/dM`` on that grid.
    class_dndm:
        Per-class ``dN/dM`` on the same grid, keyed by population class.
    path:
        Output image path; parent directories are created.
    dpi:
        Raster resolution (``diagnostics.figure_dpi``).
    class_order:
        Plot order for the class curves. Classes present in ``class_dndm`` but
        absent here are appended in sorted order, so no curve is silently
        dropped by a stale order list.
    total_label / xlabel / ylabel / title:
        Series and axis text. Units carry no slash (``docs/PLOTS.md``).
    caption:
        Optional caption rendered beneath the axes. Used to carry a mandatory
        provenance banner with the figure itself rather than only in a report
        (issue #201). Rendered at the tick-label font size, so no text on the
        figure is smaller than the caption, and wrapped to a *measured* width
        with its own reserved band, so it can never overlap the x-axis label.
        A long ``title`` is wrapped the same way instead of being clipped.
    log_x / log_y:
        Log scaling. Both default on: a compact-object mass function spans
        decades and the science question is multiplicative (``docs/PLOTS.md``).
        With ``log_y``, the y range is clipped to ``plotting.dndm_y_decades``
        below the largest sample, so a smoothly decaying truncation cannot
        stretch the axis over a hundred decades and flatten every curve.
    vlines:
        Optional labelled vertical reference lines, e.g.
        ``{"M_Ch": 1.4, "M_TOV": 2.2}``. Drawn in the threshold style.
    style:
        Resolved ``config.plotting`` style; schema defaults when omitted.

    Returns
    -------
    Path | None
        The written path, or ``None`` when no curve had any finite positive
        sample to draw (nothing is written in that case).

    Limitations
    -----------
    Non-finite samples are dropped per curve, and on a log axis non-positive
    samples are dropped too — so a class whose rate is identically zero (for
    instance a class fully removed by an ``M_Ch`` truncation) simply does not
    appear, and its absence is not annotated. This primitive draws whatever it
    is handed: it neither normalizes, rescales, nor checks that the curves came
    from real inputs.
    """
    cfg = resolve_plotting_style(style)
    grid = np.asarray(mass_grid_msun, dtype=np.float64)

    ordered: list[str] = list(class_order)
    ordered += sorted(k for k in class_dndm if k not in ordered)

    def _finite_pair(
        values: Sequence[float] | NDArray[np.floating],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != grid.shape:
            raise ValueError(
                f"curve shape {arr.shape} does not match mass grid {grid.shape}"
            )
        keep = np.isfinite(arr) & np.isfinite(grid)
        if log_y:
            keep &= arr > 0.0
        if log_x:
            keep &= grid > 0.0
        return grid[keep], arr[keep]

    series: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]] = []
    tx, ty = _finite_pair(total_dndm)
    if tx.size:
        series.append((total_label, tx, ty))
    for name in ordered:
        if name not in class_dndm:
            continue
        cx, cy = _finite_pair(class_dndm[name])
        if cx.size:
            series.append((name, cx, cy))
    if not series:
        return None

    plt = require_pyplot()
    width, height = (float(cfg.figsize_landscape[0]), float(cfg.figsize_landscape[1]))

    # Side gutter for the caption band, and the width a wrapped title may use.
    gutter = 0.18
    caption_width = width - 2.0 * gutter
    caption_lines: list[str] = []
    caption_line_height = 0.0
    band = 0.0
    if caption:
        caption_lines = wrap_text_to_inches(
            caption,
            max_width_inches=caption_width,
            font_family=cfg.font_family,
            fontsize=cfg.tick_label_fontsize,
        )
        caption_line_height = (
            float(cfg.tick_label_fontsize) * float(cfg.caption_line_spacing) / 72.0
        )
        band = caption_line_height * len(caption_lines) + 2.0 * gutter
        height += band

    title_text = title
    if title:
        title_lines = wrap_text_to_inches(
            title,
            max_width_inches=width - 1.6,  # leave room for the y-axis label column
            font_family=cfg.font_family,
            fontsize=cfg.title_fontsize,
        )
        title_text = "\n".join(title_lines)

    fig, axis = plt.subplots(figsize=(width, height))
    for index, (label, xs, ys) in enumerate(series):
        sty = series_style(index, cfg)
        axis.plot(
            xs,
            ys,
            label=label,
            color=sty["color"],
            linestyle=sty["linestyle"],
            linewidth=sty["linewidth"],
            marker=sty["marker"],
            markersize=sty["markersize"],
            markevery=max(1, xs.size // 12),
        )
    if log_x:
        axis.set_xscale("log")
    if log_y:
        axis.set_yscale("log")
        # Clip the dynamic range: a soft M_TOV truncation decays smoothly toward
        # zero, so an unclipped log axis can span >100 decades and flatten every
        # curve against the top of the frame (docs/PLOTS.md).
        axis.set_ylim(
            *dndm_log_ylim(
                max(float(np.max(ys)) for _label, _xs, ys in series),
                float(cfg.dndm_y_decades),
            )
        )
    for name, value in (vlines or {}).items():
        if not np.isfinite(value):
            continue
        axis.axvline(
            float(value),
            color=cfg.threshold_color,
            linestyle=cfg.threshold_linestyle,
            linewidth=cfg.line_width,
            label=f"{name}={value:g}" + r" M$_{\odot}$",
        )
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title_text)
    axis.legend(
        loc="best",
        fontsize=cfg.legend_fontsize,
        prop={"family": cfg.font_family},
        ncols=2 if len(series) > 4 else 1,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not caption_lines:
        return save_figure(fig, path, dpi=dpi)

    # Reserve the caption band explicitly rather than relying on tight_layout,
    # which would let long caption text overlap the x-axis label.
    band_fraction = band / height
    fig.tight_layout(rect=(0.0, band_fraction, 1.0, 1.0))
    fig.text(
        gutter / width,
        (band - gutter) / height,
        "\n".join(caption_lines),
        ha="left",
        va="top",
        fontfamily=cfg.font_family,
        fontsize=cfg.tick_label_fontsize,
        linespacing=float(cfg.caption_line_spacing),
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_m2_posterior_convergence(
    probabilities: Sequence[float] | NDArray[np.floating],
    sigmas: Sequence[float] | NDArray[np.floating],
    path: Path,
    *,
    probability_cut: float,
    dpi: int,
    xlabel: str = r"$P(M_2 > M_{\mathrm{lim}})$",
    ylabel: str = r"MC $\sigma_P$",
    title: str = "m2_posterior_convergence",
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Scatter of per-system probability vs binomial MC error, with the cut line."""
    pp = np.asarray(probabilities, dtype=np.float64)
    ss = np.asarray(sigmas, dtype=np.float64)
    finite = np.isfinite(pp) & np.isfinite(ss)
    if not np.any(finite):
        return None
    cfg = resolve_plotting_style(style)
    sty = series_style(0, cfg)
    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))
    axis.scatter(
        pp[finite],
        ss[finite],
        marker=sty["marker"],
        s=max(sty["markersize"] ** 2, 16.0),
        color=sty["color"],
        linewidths=0.0,
    )
    axis.axvline(
        float(probability_cut),
        color=cfg.threshold_color,
        linestyle=cfg.threshold_linestyle,
        linewidth=cfg.line_width,
        label=f"cut={probability_cut:g}",
    )
    axis.legend(
        loc="best",
        fontsize=cfg.legend_fontsize,
        prop={"family": cfg.font_family},
    )
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
    return save_figure(fig, path, dpi=dpi)
