"""Shared rendering primitives for stage diagnostics and paper-ready product figures.

ARCHITECTURE.md §4. Matplotlib is an optional dependency (``pip install -e ".[plot]"``);
callers must tolerate missing matplotlib when figures are disabled or unavailable.
Display-only: excluded from stage ``source_hash`` dependency lists.

Style defaults live in ``config.plotting`` / ``PlottingStyleConfig`` (see ``docs/PLOTS.md``).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop.config_schema import PlottingStyleConfig

#: Axis scale a primitive may choose for itself (``choose_axis_scale``).
AxisScale = Literal["linear", "log", "symlog"]

#: Column / quantity key → axis label with units (``docs/PLOTS.md``, "Always
#: label axes with units"; #333). Display text only: no physics lives here.
#: ``axis_label`` falls back to the key itself for anything not listed, so a
#: missing entry shows up as a raw key rather than a wrong unit.
AXIS_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ruwe": "RUWE",
        "period_day": r"$P_{\rm orb}$ (day)",
        "P_orb_days": r"$P_{\rm orb}$ (day)",
        "eccentricity": "eccentricity",
        "g_mag": r"$G$ (mag)",
        "G_mag": r"$G$ (mag)",
        "m1_msun": r"$M_1$ (M$_\odot$)",
        "m2_msun": r"$M_2$ (M$_\odot$)",
        "m1_tilde_msun": r"$\tilde{M}_1$ (M$_\odot$)",
        "m2_tilde_msun": r"$\tilde{M}_2$ (M$_\odot$)",
        "companion_mass_msun": r"companion mass (M$_\odot$)",
        # f_m is the binary mass function, not a mass fraction (#333).
        "f_m_msun": r"mass function $f_m$ (M$_\odot$)",
        "inv_parallax_mas_inv": r"$1/\varpi$ (kpc)",
        "parallax_mas": r"$\varpi$ (mas)",
        "cos_inclination": r"$\cos i$",
        "ra_deg": "right ascension (deg)",
        "dec_deg": "declination (deg)",
        "chi2_dof": r"$\chi^2$ per degree of freedom",
        "delta_bic_wd_vs_dark": r"$\Delta{\rm BIC}$ (WD $-$ dark)",
        "logz": r"$\ln Z$",
        "n_mock": r"$N_{\rm mock}$",
        "mc_poisson_ratio": r"max $\sigma_{\rm MC} / \sigma_{\rm Poisson}$",
        "survival_probability": "survival probability",
        "source_id": "Gaia DR3 source_id",
        "cut_id": "cut",
        "fit_tier": "fit tier",
        "age_gyr": "primary age (Gyr)",
        "count": "count",
    }
)


def axis_label(name: str) -> str:
    """Axis label with units for a column / quantity key (#333).

    Parameters
    ----------
    name:
        A data key such as ``"m2_msun"`` or ``"period_day"``, or text that is
        already a label.

    Returns
    -------
    str
        The ``AXIS_LABELS`` entry when ``name`` is a known key; otherwise
        ``name`` unchanged.

    Limitations
    -----------
    Unknown keys pass through verbatim, so a raw key on a figure means the
    mapping needs an entry — never a guessed unit.
    """
    return AXIS_LABELS.get(name, name)


def math_fontfamily(style: PlottingStyleConfig | None = None) -> str:
    """Mathtext font set matching ``plotting.font_family`` (serif text, serif math).

    Matplotlib renders ``$...$`` in DejaVu Sans regardless of the text family,
    so serif labels otherwise mix two typefaces (``docs/PLOTS.md``: serif).
    """
    cfg = resolve_plotting_style(style)
    family = cfg.font_family.lower()
    return "dejavuserif" if "serif" in family and "sans" not in family else "dejavusans"


def legend_prop(style: PlottingStyleConfig | None = None) -> dict[str, Any]:
    """Legend font properties: family **and** size.

    Matplotlib ignores ``legend(fontsize=...)`` whenever ``prop=`` is also given,
    so the size must live inside ``prop`` (#360).
    """
    cfg = resolve_plotting_style(style)
    return {
        "family": cfg.font_family,
        "size": float(cfg.legend_fontsize),
        "math_fontfamily": math_fontfamily(cfg),
    }


def reference_line_style(
    index: int,
    style: PlottingStyleConfig | None = None,
) -> dict[str, Any]:
    """Colour / linestyle / width for labelled reference line ``index``.

    Colours and linestyles cycle together from ``plotting.reference_line_colors``
    and ``plotting.reference_linestyle_cycle``, so two reference lines on one
    panel (``M_Ch`` and ``M_TOV``) never share a style (#333).
    """
    cfg = resolve_plotting_style(style)
    colors = list(cfg.reference_line_colors) or [cfg.threshold_color]
    linestyles = list(cfg.reference_linestyle_cycle) or [cfg.threshold_linestyle]
    return {
        "color": colors[index % len(colors)],
        "linestyle": linestyles[index % len(linestyles)],
        "linewidth": float(cfg.line_width),
    }


def choose_axis_scale(
    values: NDArray[np.floating] | Sequence[float],
    *,
    min_decades: float,
) -> AxisScale:
    """Pick linear / log / symlog for a distribution from its dynamic range.

    Parameters
    ----------
    values:
        Data to be drawn along the axis. Non-finite entries are ignored.
    min_decades:
        Threshold in decades (``plotting.auto_log_min_decades``).

    Returns
    -------
    AxisScale
        ``"log"`` when every finite value is positive and
        ``log10(max / min) >= min_decades``; ``"symlog"`` when the data contain
        zeros or negatives and ``log10(max|x| / median|x|) >= min_decades``
        (median over non-zero ``|x|``); otherwise ``"linear"``.

    Limitations
    -----------
    A purely range-based rule: it cannot tell whether the science question is
    multiplicative (``docs/PLOTS.md``). Callers that know better pass an explicit
    scale. Fewer than two finite values always give ``"linear"``.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return "linear"
    if np.all(arr > 0.0):
        span = math.log10(float(np.max(arr)) / float(np.min(arr)))
        return "log" if span >= float(min_decades) else "linear"
    magnitude = np.abs(arr)
    nonzero = magnitude[magnitude > 0.0]
    if nonzero.size == 0:
        return "linear"
    span = math.log10(float(np.max(nonzero)) / float(np.median(nonzero)))
    return "symlog" if span >= float(min_decades) else "linear"


def symlog_linthresh(values: NDArray[np.floating] | Sequence[float]) -> float:
    """Linear-region half-width for a symlog axis: median of non-zero ``|x|``.

    Returns ``1.0`` when no finite non-zero value exists.
    """
    arr = np.abs(np.asarray(values, dtype=np.float64))
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 1.0
    return float(np.median(arr))


def _scale_transforms(
    scale: AxisScale, linthresh: float
) -> tuple[
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
]:
    """Forward / inverse coordinate maps matching matplotlib's axis scales."""
    if scale == "log":
        return np.log10, lambda t: np.power(10.0, t)
    if scale == "symlog":
        from matplotlib.scale import SymmetricalLogTransform

        forward = SymmetricalLogTransform(10.0, float(linthresh), 1.0)
        inverse = forward.inverted()
        return (
            lambda x: np.asarray(forward.transform(np.asarray(x)), dtype=np.float64),
            lambda t: np.asarray(inverse.transform(np.asarray(t)), dtype=np.float64),
        )
    return (lambda x: x), (lambda t: t)


def scaled_histogram_edges(
    finite: NDArray[np.floating],
    *,
    scale: AxisScale,
    bins: int | str = "auto",
    max_bins: int | None = None,
    linthresh: float = 1.0,
) -> NDArray[np.float64]:
    """Histogram bin edges uniform in the plotted coordinate.

    Bins are uniform in ``x`` (linear), ``log10 x`` (log), or the symlog
    coordinate, so bars have equal visual width on the chosen axis. The bin
    count follows ``bins`` in that coordinate, capped by ``max_bins``
    (``diagnostics.histogram_max_bins``).

    Parameters
    ----------
    finite:
        Finite values only; for ``scale="log"`` they must also be positive.
    scale / linthresh:
        Axis scale and, for symlog, its linear half-width.
    bins / max_bins:
        As for ``resolve_histogram_bins``.
    """
    arr = np.asarray(finite, dtype=np.float64)
    forward, inverse = _scale_transforms(scale, linthresh)
    coord = forward(arr)
    resolved = resolve_histogram_bins(coord, bins, max_bins=max_bins)
    edges = np.histogram_bin_edges(coord, bins=resolved)
    return np.asarray(inverse(edges), dtype=np.float64)


#: Most major ticks a symlog axis gets; decades are thinned to stay under it
#: so the ``±10^k`` labels never collide (#333).
_SYMLOG_MAX_TICKS: Final[int] = 7


def symlog_major_ticks(
    lo: float, hi: float, linthresh: float, *, max_ticks: int = _SYMLOG_MAX_TICKS
) -> list[float]:
    """Thinned ``0, ±10^k`` major ticks for a symlog axis spanning ``[lo, hi]``.

    Decades start at the first power of ten at or above ``linthresh``; every
    ``step``-th decade is kept (same ``k`` on both signs) so at most
    ``max_ticks`` ticks are returned. Zero is always included.
    """
    k_min = math.ceil(math.log10(max(float(linthresh), 1e-300)))
    k_neg = math.floor(math.log10(-lo)) if lo < 0.0 and -lo >= 10.0**k_min else None
    k_pos = math.floor(math.log10(hi)) if hi > 0.0 and hi >= 10.0**k_min else None
    decades: list[tuple[int, int]] = []  # (sign, k)
    if k_neg is not None:
        decades += [(-1, k) for k in range(k_min, k_neg + 1)]
    if k_pos is not None:
        decades += [(1, k) for k in range(k_min, k_pos + 1)]
    n_slots = max(1, int(max_ticks) - 1)
    step = max(1, math.ceil(len(decades) / n_slots))
    kept = [sign * 10.0**k for sign, k in decades if (k - k_min) % step == 0]
    return sorted(kept + [0.0])


def _apply_axis_scale(
    axis: Any, which: Literal["x", "y"], scale: AxisScale, linthresh: float
) -> None:
    setter = axis.set_xscale if which == "x" else axis.set_yscale
    if scale == "symlog":
        from matplotlib.ticker import FixedLocator

        setter("symlog", linthresh=float(linthresh))
        lo, hi = axis.get_xlim() if which == "x" else axis.get_ylim()
        ticks = symlog_major_ticks(float(lo), float(hi), float(linthresh))
        (axis.xaxis if which == "x" else axis.yaxis).set_major_locator(
            FixedLocator(ticks)
        )
    elif scale == "log":
        setter("log")


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

    Curved projections (Mollweide, Aitoff, polar, ...) have no top/right edge to
    tick, and inward ticks there land inside the map frame (#333): they get
    fonts, label sizes and spine widths only.
    """
    cfg = resolve_plotting_style(style)
    if getattr(axis, "name", "rectilinear") != "rectilinear":
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
        axis.tick_params(axis="both", which="both", labelsize=cfg.tick_label_fontsize)
        for spine in axis.spines.values():
            spine.set_linewidth(cfg.spines_width)
        for label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
            label.set_fontfamily(cfg.font_family)
            label.set_math_fontfamily(math_fontfamily(cfg))
        for text in (axis.xaxis.label, axis.yaxis.label, axis.title):
            text.set_math_fontfamily(math_fontfamily(cfg))
        return
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
    for text in (axis.xaxis.label, axis.yaxis.label, axis.title):
        text.set_math_fontfamily(math_fontfamily(cfg))

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
        label.set_math_fontfamily(math_fontfamily(cfg))


#: Tolerance (inches) when testing whether drawn text spills past the canvas.
_OVERFLOW_TOLERANCE_INCHES: Final[float] = 1e-3


def _math_aware_tokens(text: str) -> list[str]:
    """Split on whitespace, keeping ``$...$`` mathtext (which may contain spaces) whole."""
    return re.findall(r"(?:\$[^$]*\$|[^\s$])+", text)


def wrap_label_to_inches(
    text: str,
    *,
    max_width_inches: float,
    font_family: str,
    fontsize: float,
) -> str:
    """Greedy-wrap a title or label to a measured width, mathtext-safe.

    Unlike ``wrap_text_to_inches`` (captions), this never breaks inside a
    ``$...$`` span, so labels such as ``r"$M_{\\rm Ch}$"`` stay parseable.
    Existing newlines are treated as spaces and the text is re-flowed.

    Limitations
    -----------
    A single token wider than the budget stays on its own over-wide line.
    O(tokens) renderer calls; meant for titles, not bulk tick labels.
    """
    tokens = _math_aware_tokens(text)
    if not tokens or max_width_inches <= 0.0:
        return text
    lines: list[str] = []
    current = tokens[0]
    for token in tokens[1:]:
        candidate = f"{current} {token}"
        if (
            measure_text_width_inches(
                candidate, font_family=font_family, fontsize=fontsize
            )
            <= max_width_inches
        ):
            current = candidate
        else:
            lines.append(current)
            current = token
    lines.append(current)
    return "\n".join(lines)


def _fit_titles(fig: Any) -> bool:
    """Wrap every axes title to its axes width (suptitle to the figure width).

    Returns True when any title text changed.
    """
    changed = False
    fig_width = float(fig.get_figwidth())
    targets: list[tuple[Any, float]] = []
    for axis in fig.axes:
        if not axis.get_visible():
            continue
        targets.append((axis.title, float(axis.get_position().width) * fig_width))
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        targets.append((suptitle, fig_width))
    for artist, budget in targets:
        text = artist.get_text()
        if not text:
            continue
        family = artist.get_fontfamily()
        wrapped = wrap_label_to_inches(
            text,
            max_width_inches=budget,
            font_family=family[0] if isinstance(family, list) else str(family),
            fontsize=float(artist.get_fontsize()),
        )
        if wrapped != text:
            artist.set_text(wrapped)
            changed = True
    return changed


def layout_figure(fig: Any) -> None:
    """Wrap titles to their axes and apply ``tight_layout`` (#333: no clipped text).

    Titles are wrapped before the first layout pass (so an over-long title does
    not distort it) and again against the post-layout axes widths.
    """
    import warnings

    _fit_titles(fig)
    with warnings.catch_warnings():
        # tight_layout warns (and leaves the layout alone) for axes it cannot fit;
        # figure_overflows / save_figure catch any resulting spill.
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
        if _fit_titles(fig):
            fig.tight_layout()


def figure_overflows(fig: Any) -> bool:
    """True when any drawn artist extends past the figure canvas (would be clipped)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    width, height = (float(v) for v in fig.get_size_inches())
    tol = _OVERFLOW_TOLERANCE_INCHES
    return bool(
        bbox.x0 < -tol
        or bbox.y0 < -tol
        or bbox.x1 > width + tol
        or bbox.y1 > height + tol
    )


def save_figure(fig: Any, path: Path, *, dpi: int) -> Path:
    """Lay out, save to ``path`` (parents created) and close a figure.

    Layout is ``layout_figure`` (wrapped titles + ``tight_layout``). If any text
    still spills past the canvas, the figure is saved with
    ``bbox_inches="tight"`` so the canvas grows to hold it rather than clipping
    it (#333); otherwise the configured figure size is kept exactly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layout_figure(fig)
    bbox = "tight" if figure_overflows(fig) else None
    fig.savefig(path, dpi=dpi, bbox_inches=bbox)
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
    log_x: bool | Literal["auto", "symlog"] = "auto",
    caption: str | None = None,
) -> Path | None:
    """Write a one-dimensional histogram PNG. Returns None when values are empty.

    Non-finite entries are dropped so ``bins="auto"`` does not raise on NaN ranges
    (e.g. missing eccentricities in NSS orbital blocks). When ``xlim`` is set, only
    values inside ``[xlim[0], xlim[1]]`` are binned and the axis is fixed; counts
    outside the range are appended to the title so heavy tails are not hidden.

    ``log_x`` (ignored when ``xlim`` is set, which keeps a fixed linear window):
    ``"auto"`` lets ``choose_axis_scale`` pick linear / log / symlog from the
    data's dynamic range (``plotting.auto_log_min_decades``), so heavy-tailed
    quantities (RUWE, period, chi2/dof, ΔBIC) are not squeezed into the first
    few bins (#333); ``True`` forces log (non-positive values are dropped and
    counted in the title); ``"symlog"`` forces symlog; ``False`` forces linear.
    Bins are uniform in the plotted coordinate and capped by ``max_bins``.

    ``caption`` is wrapped into a band reserved beneath the axes at tick-label
    size, as for ``plot_dndm_by_class``.
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
    width, height = (float(cfg.figsize_landscape[0]), float(cfg.figsize_landscape[1]))
    caption_lines, band = _caption_band(cfg, width, caption)
    height += band
    fig, axis = plt.subplots(figsize=(width, height))

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
        scale: AxisScale
        if log_x == "auto":
            scale = choose_axis_scale(
                finite, min_decades=float(cfg.auto_log_min_decades)
            )
        elif log_x == "symlog":
            scale = "symlog"
        else:
            scale = "log" if log_x else "linear"
        drawn = finite
        if scale == "log":
            drawn = finite[finite > 0.0]
            n_nonpositive = int(finite.size - drawn.size)
            if drawn.size == 0:
                plt.close(fig)
                return None
            if n_nonpositive:
                display_title = f"{title} ({n_nonpositive} values $\\leq 0$ omitted)"
        linthresh = symlog_linthresh(drawn)
        if scale == "linear":
            resolved: int | str | NDArray[np.floating] = resolve_histogram_bins(
                drawn, bins, max_bins=max_bins
            )
        else:
            resolved = scaled_histogram_edges(
                drawn, scale=scale, bins=bins, max_bins=max_bins, linthresh=linthresh
            )
        axis.hist(
            drawn,
            bins=resolved,
            color=face,
            edgecolor=cfg.hist_edge_color,
            linewidth=cfg.spines_width * 0.5,
        )
        _apply_axis_scale(axis, "x", scale, linthresh)

    if log_y:
        axis.set_yscale("log")
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=display_title)
    if not caption_lines:
        return save_figure(fig, path, dpi=dpi)
    return _save_with_caption_band(
        fig, path, dpi=dpi, cfg=cfg, caption_lines=caption_lines, band=band
    )


#: Mollweide graticule: RA meridians and Dec parallels (degrees) that get labels.
_SKY_RA_TICKS_DEG: Final[tuple[float, ...]] = (-120.0, -60.0, 0.0, 60.0, 120.0)
_SKY_DEC_TICKS_DEG: Final[tuple[float, ...]] = (-60.0, -30.0, 0.0, 30.0, 60.0)


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
    xlabel: str = AXIS_LABELS["ra_deg"],
    ylabel: str = AXIS_LABELS["dec_deg"],
) -> Path | None:
    """Write an equatorial Mollweide sky map. Returns None when coordinates are empty.

    Marker size / alpha are usually supplied from ``diagnostics.sky_map_*``; when
    omitted, a modest default suitable for small samples is used. Axes carry
    RA / Dec labels with units and a coordinate grid; no edge ticks are drawn
    inside the map frame (#333).
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
    # Matplotlib's default 30 deg RA labels collide along the equator at
    # tick-label size; 60 / 30 deg spacing keeps every label legible.
    ax.set_xticks(np.radians(_SKY_RA_TICKS_DEG))
    ax.set_yticks(np.radians(_SKY_DEC_TICKS_DEG))
    ax.grid(True, color=cfg.hist_edge_color, alpha=0.3, linewidth=cfg.line_width * 0.5)
    apply_axes_style(
        ax,
        cfg,
        xlabel=xlabel,
        ylabel=ylabel,
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
    log_x: bool | Literal["auto"] = "auto",
) -> Path | None:
    """Overlay named histograms (El-Badry-style mock vs real panel building block).

    ``log_x="auto"`` chooses a log x axis when the pooled finite data are
    positive and span ``plotting.auto_log_min_decades`` or more (#333); every
    series then shares one set of log-spaced edges and, with ``density``, is
    normalised per dex (the label ``"density"`` becomes ``density (dex^-1)``).
    ``True`` forces log (non-positive values dropped), ``False`` linear.
    """
    cfg = resolve_plotting_style(style)
    prepared: list[tuple[str, NDArray[np.float64]]] = []
    for label, values in series.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            prepared.append((label, finite))
    if not prepared:
        return None

    pooled = np.concatenate([arr for _label, arr in prepared])
    if log_x == "auto":
        use_log = (
            choose_axis_scale(pooled, min_decades=float(cfg.auto_log_min_decades))
            == "log"
        )
    else:
        use_log = bool(log_x)
    shared_edges: NDArray[np.float64] | None = None
    if use_log:
        prepared = [(lbl, a[a > 0.0]) for lbl, a in prepared if np.any(a > 0.0)]
        if not prepared:
            return None
        pooled = np.concatenate([arr for _label, arr in prepared])
        shared_edges = scaled_histogram_edges(
            pooled, scale="log", bins=bins, max_bins=max_bins
        )
        if density and ylabel == "density":
            ylabel = r"density (dex$^{-1}$)"

    plt = require_pyplot()
    fig, axis = plt.subplots(figsize=tuple(cfg.figsize_landscape))
    for index, (label, arr) in enumerate(prepared):
        sty = series_style(index, cfg)
        if shared_edges is None:
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
            continue
        counts, edges = np.histogram(arr, bins=shared_edges)
        heights = counts.astype(np.float64)
        if density and counts.sum() > 0:
            heights = heights / (counts.sum() * np.diff(np.log10(edges)))
        axis.stairs(
            heights,
            edges,
            linewidth=sty["linewidth"],
            label=label,
            color=sty["color"],
            linestyle=sty["linestyle"],
        )
    if shared_edges is not None:
        axis.set_xscale("log")
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
    axis.legend(loc="best", prop=legend_prop(cfg))
    return save_figure(fig, path, dpi=dpi)


def watermark_png(path: Path, text: str) -> Path:
    """Stamp ``text`` diagonally across an existing PNG, in place (#355 demo marking).

    The image is re-rendered at its native pixel size; the stamp is large,
    semi-transparent and centred so it cannot be mistaken for data.
    """
    plt = require_pyplot()
    path = Path(path)
    img = plt.imread(path)
    height, width = img.shape[0], img.shape[1]
    dpi = 100.0
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(img)
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        rotation=25,
        fontsize=max(12.0, width / dpi * 9.0),
        color="#D55E00",
        alpha=0.45,
        fontweight="bold",
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def six_panel_bin_edges(
    xmin: float,
    xmax: float,
    *,
    scale: str,
    n_bins: int,
) -> NDArray[np.float64]:
    """Fixed histogram edges over ``[xmin, xmax]``: linear, or log-spaced for ``log``."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if scale == "log":
        if xmin <= 0:
            raise ValueError("log-scale bin edges need xmin > 0")
        return np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
    if scale != "linear":
        raise ValueError(f"unsupported scale {scale!r}")
    return np.linspace(xmin, xmax, n_bins + 1)


def plot_six_panel_grid(
    panels: Mapping[str, Mapping[str, NDArray[np.floating] | Sequence[float]]],
    path: Path,
    *,
    panel_order: Sequence[str],
    dpi: int,
    title: str = "El-Badry-style six-panel comparison",
    panel_xlabels: Mapping[str, str] | None = None,
    panel_axes: Mapping[str, tuple[str, float, float]] | None = None,
    panel_ylabels: Mapping[str, str] | None = None,
    bins: int | str = "auto",
    max_bins: int | None = None,
    density: bool = True,
    caption: str | None = None,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Write a 2×3 grid of overlay histograms (selection-function validation style).

    ``panels`` maps panel name → {series_label → values}. Missing or empty series are
    skipped within a panel; a panel with no data is left blank with an annotation.

    ``panel_axes`` maps panel name → ``(scale, xmin, xmax)`` with ``scale`` in
    ``{"linear", "log"}``. Such a panel uses fixed edges over that range
    (``max_bins`` or 40 bins, log-spaced on a log axis); values outside it are not
    drawn, so the caller should report out-of-range counts. Other panels keep
    ``bins`` (capped by ``max_bins``) over their finite data. ``density=True``
    normalises each drawn series to unit area **in the plotted coordinate** — per
    unit x on a linear axis, per dex on a log axis — so samples of very different
    size compare by shape and a log panel is not skewed toward wide bins.
    ``caption`` is wrapped into its own reserved band beneath the axes at
    tick-label size (docs/PLOTS.md).
    """
    if not panel_order:
        return None
    cfg = resolve_plotting_style(style)
    plt = require_pyplot()
    n = len(panel_order)
    nrows = 2 if n > 3 else 1
    ncols = min(3, n) if n else 1
    width = cfg.figsize_landscape[0] / 7.0 * 4 * ncols
    height = 3.5 * nrows
    gutter = 0.18
    caption_lines: list[str] = []
    band = 0.0
    if caption:
        caption_lines = wrap_text_to_inches(
            caption,
            max_width_inches=width - 2.0 * gutter,
            font_family=cfg.font_family,
            fontsize=cfg.tick_label_fontsize,
        )
        line_h = float(cfg.tick_label_fontsize) * float(cfg.caption_line_spacing) / 72.0
        band = line_h * len(caption_lines) + 2.0 * gutter
        height += band
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    flat = np.atleast_1d(axes).ravel()
    any_drawn = False
    for index, panel_name in enumerate(panel_order):
        axis = flat[index]
        series = panels.get(panel_name, {})
        axis_spec = (panel_axes or {}).get(panel_name)
        drawn = False
        series_index = 0
        for label, values in series.items():
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if axis_spec is not None:
                scale, lo, hi = axis_spec
                finite = finite[(finite >= lo) & (finite <= hi)]
                resolved: int | str | NDArray[np.floating] = six_panel_bin_edges(
                    lo, hi, scale=scale, n_bins=int(max_bins) if max_bins else 40
                )
            if finite.size == 0:
                continue
            if axis_spec is None:
                resolved = resolve_histogram_bins(finite, bins, max_bins=max_bins)
            sty = series_style(series_index, cfg)
            counts, edges = np.histogram(finite, bins=resolved)
            heights = counts.astype(np.float64)
            if density and counts.sum() > 0:
                coord = (
                    np.log10(edges)
                    if axis_spec is not None and axis_spec[0] == "log"
                    else edges
                )
                heights = heights / (counts.sum() * np.diff(coord))
            axis.stairs(
                heights,
                edges,
                linewidth=sty["linewidth"],
                label=label,
                color=sty["color"],
                linestyle=sty["linestyle"],
            )
            series_index += 1
            drawn = True
            any_drawn = True
        if axis_spec is not None:
            scale, lo, hi = axis_spec
            if scale == "log":
                axis.set_xscale("log")
            axis.set_xlim(lo, hi)
        apply_axes_style(
            axis,
            cfg,
            xlabel=(panel_xlabels or {}).get(panel_name, panel_name),
            ylabel=(panel_ylabels or {}).get(
                panel_name,
                (
                    "density (dex$^{-1}$)"
                    if axis_spec is not None and axis_spec[0] == "log"
                    else "density"
                )
                if density
                else "count",
            ),
        )
        if drawn:
            axis.legend(loc="best", prop=legend_prop(cfg))
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
    if not caption_lines:
        return save_figure(fig, path, dpi=dpi)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, band / height, 1.0, 1.0))
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


#: Fraction of the figure width the value axis gets on a vertical bar chart,
#: used to estimate whether category labels fit their slots before layout.
_AXES_WIDTH_FRACTION: Final[float] = 0.75
#: Share of one category slot a label may fill before bars go horizontal.
_LABEL_SLOT_FILL: Final[float] = 0.9
#: Share of a category slot covered by its bar(s); the rest is the gap.
_BAR_GROUP_FILL: Final[float] = 0.8
#: Inches added to a horizontal bar chart for title, value label and ticks.
_BAR_FIGURE_OVERHEAD_INCHES: Final[float] = 1.8
#: Minimum inches given to the value axis of a horizontal bar chart.
_BAR_VALUE_AXIS_INCHES: Final[float] = 6.0
#: Inches a horizontal bar chart spends beside the axes besides tick labels
#: (category-axis label, paddings), used to predict the post-layout axes width.
_BAR_SIDE_MARGIN_INCHES: Final[float] = 1.4
#: Decades below the smallest positive bar where a log value axis starts, so
#: the smallest bar has visible length and zero-valued bars sit at the floor.
_LOG_BAR_FLOOR_DECADES: Final[float] = 0.5
#: Safety factor on the measured annotation size when reserving headroom.
_ANNOTATION_HEADROOM_FACTOR: Final[float] = 1.5


def format_bar_value(value: float) -> str:
    """Annotation text for one bar: integers without decimals, ``n/a`` for NaN."""
    if not math.isfinite(value):
        return "n/a"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):d}"
    return f"{value:.4g}"


def bars_should_be_horizontal(
    labels: Sequence[str],
    style: PlottingStyleConfig | None = None,
) -> bool:
    """True when category labels would overlap on a vertical bar chart (#333).

    Horizontal when there are more than ``plotting.categorical_max_vertical_labels``
    categories, or when any label, measured at ``tick_label_fontsize``, is wider
    than ``_LABEL_SLOT_FILL`` of its slot on a ``figsize_landscape`` figure.
    """
    cfg = resolve_plotting_style(style)
    if not labels:
        return False
    if len(labels) > int(cfg.categorical_max_vertical_labels):
        return True
    slot = float(cfg.figsize_landscape[0]) * _AXES_WIDTH_FRACTION / len(labels)
    widest = max(
        measure_text_width_inches(
            str(label), font_family=cfg.font_family, fontsize=cfg.tick_label_fontsize
        )
        for label in labels
    )
    return widest > _LABEL_SLOT_FILL * slot


def choose_bar_value_scale(
    values: Sequence[float] | NDArray[np.floating],
    style: PlottingStyleConfig | None = None,
) -> Literal["linear", "log"]:
    """Log value axis when non-negative bar heights span many decades (#333).

    Log when every finite value is ``>= 0``, at least two are positive, and
    ``log10(max / min_positive) >= plotting.auto_log_min_decades`` — e.g. gate
    outcomes of 48 / 27 against 7741 skipped. Zero-valued bars then sit at the
    axis floor, and their annotation still reads ``0``.
    """
    cfg = resolve_plotting_style(style)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.any(arr < 0.0):
        return "linear"
    positive = arr[arr > 0.0]
    if positive.size < 2:
        return "linear"
    span = math.log10(float(np.max(positive)) / float(np.min(positive)))
    return "log" if span >= float(cfg.auto_log_min_decades) else "linear"


def _outside_legend_inches(
    names: Sequence[str | None], horizontal: bool, cfg: PlottingStyleConfig
) -> float:
    """Width an outside-right legend takes (grouped horizontal bars only)."""
    shown = [n for n in names if n]
    if not horizontal or len(shown) < 2:
        return 0.0
    return _LEGEND_HANDLE_INCHES + max(
        measure_text_width_inches(
            n, font_family=cfg.font_family, fontsize=cfg.legend_fontsize
        )
        for n in shown
    )


def _bar_figure_size(
    labels: Sequence[str],
    n_series: int,
    horizontal: bool,
    cfg: PlottingStyleConfig,
    legend_inches: float = 0.0,
) -> tuple[float, float]:
    """Figure size that keeps every label and bar at configured font sizes."""
    width, height = (float(cfg.figsize_landscape[0]), float(cfg.figsize_landscape[1]))
    if not horizontal:
        return width, height
    widest = max(
        measure_text_width_inches(
            str(label), font_family=cfg.font_family, fontsize=cfg.tick_label_fontsize
        )
        for label in labels
    )
    rows = len(labels) * max(1, n_series)
    height = max(
        height,
        rows * float(cfg.categorical_row_height_inches) + _BAR_FIGURE_OVERHEAD_INCHES,
    )
    width = max(width, widest + _BAR_VALUE_AXIS_INCHES + legend_inches)
    return width, height


def _draw_bar_series(
    fig: Any,
    axis: Any,
    labels: Sequence[str],
    series: Sequence[tuple[str | None, NDArray[np.float64], str, str]],
    *,
    horizontal: bool,
    scale: Literal["linear", "log"],
    annotate: bool,
    cfg: PlottingStyleConfig,
) -> None:
    """Draw one or more bar series on ``axis`` with readable labels and values.

    NaN heights draw no bar and are annotated ``n/a``. With ``annotate``, every
    bar carries its value at ``tick_label_fontsize`` and the value axis gets
    measured headroom so the text stays inside the frame.
    """
    n_cat = len(labels)
    n_series = max(1, len(series))
    thickness = _BAR_GROUP_FILL / n_series
    positions = np.arange(n_cat, dtype=np.float64)
    all_values = (
        np.concatenate([vals for _n, vals, _c, _h in series]) if series else np.array([])
    )
    finite = all_values[np.isfinite(all_values)]
    positive = finite[finite > 0.0]

    if scale == "log" and positive.size:
        floor = float(np.min(positive)) * 10.0 ** (-_LOG_BAR_FLOOR_DECADES)
    else:
        floor = min(0.0, float(np.min(finite))) if finite.size else 0.0
    ceiling = float(np.max(finite)) if finite.size else 1.0
    if ceiling <= floor:
        ceiling = floor + 1.0

    texts: list[tuple[float, float, str, bool]] = []
    for index, (name, values, color, hatch) in enumerate(series):
        offset = (index - 0.5 * (n_series - 1)) * thickness
        heights = np.where(np.isfinite(values), values, 0.0)
        if scale == "log":
            # Bars on a log axis grow from the floor, not from zero.
            draw = np.where(heights > 0.0, heights - floor, 0.0)
            base = floor
        else:
            draw = heights
            base = 0.0
        kwargs: dict[str, Any] = dict(
            color=color,
            edgecolor=cfg.hist_edge_color,
            linewidth=cfg.spines_width * 0.5,
            hatch=hatch or None,
            label=name,
        )
        if horizontal:
            axis.barh(positions + offset, draw, height=thickness, left=base, **kwargs)
        else:
            axis.bar(positions + offset, draw, width=thickness, bottom=base, **kwargs)
        for pos, value in zip(positions + offset, values):
            anchor = float(value) if np.isfinite(value) else base
            if scale == "log" and anchor <= 0.0:
                anchor = floor
            at_floor = not (np.isfinite(value) and value > base)
            texts.append((float(pos), anchor, format_bar_value(float(value)), at_floor))

    tick_labels = [str(label) for label in labels]
    if horizontal:
        axis.set_yticks(positions)
        axis.set_yticklabels(
            tick_labels, fontfamily=cfg.font_family, fontsize=cfg.tick_label_fontsize
        )
        axis.set_ylim(n_cat - 0.5, -0.5)  # first category at the top
    else:
        axis.set_xticks(positions)
        axis.set_xticklabels(
            tick_labels, fontfamily=cfg.font_family, fontsize=cfg.tick_label_fontsize
        )
        axis.set_xlim(-0.5, n_cat - 0.5)

    if scale == "log":
        (axis.set_xscale if horizontal else axis.set_yscale)("log")

    # Headroom: the largest annotation, measured, as a fraction of the value
    # axis length expected after layout (tick labels take the rest).
    position = axis.get_position()
    if horizontal:
        widest_label = max(
            measure_text_width_inches(
                label, font_family=cfg.font_family, fontsize=cfg.tick_label_fontsize
            )
            for label in tick_labels
        )
        axis_inches = (
            float(fig.get_figwidth())
            - widest_label
            - _BAR_SIDE_MARGIN_INCHES
            - _outside_legend_inches([n for n, _v, _c, _h in series], horizontal, cfg)
        )
        text_inches = max(
            (
                measure_text_width_inches(
                    t, font_family=cfg.font_family, fontsize=cfg.tick_label_fontsize
                )
                for _p, _a, t, _f in texts
            ),
            default=0.0,
        )
    else:
        axis_inches = float(position.height) * float(fig.get_figheight())
        text_inches = float(cfg.tick_label_fontsize) / 72.0
    frac = 0.0
    if annotate and axis_inches > 0.0:
        frac = min(0.45, _ANNOTATION_HEADROOM_FACTOR * text_inches / axis_inches)
    if scale == "log":
        lo, hi = math.log10(floor), math.log10(ceiling)
        hi += (hi - lo) * frac / (1.0 - frac)
        limits = (floor, 10.0**hi)
    else:
        span = ceiling - floor
        limits = (floor, ceiling + span * frac / (1.0 - frac))
    (axis.set_xlim if horizontal else axis.set_ylim)(*limits)

    if annotate:
        # Values sitting at the axis floor clear the inward tick marks.
        floor_pad = float(cfg.tick_major_length) + 3.0
        for pos, anchor, text, at_floor in texts:
            pad = floor_pad if at_floor else 3.0
            if horizontal:
                axis.annotate(
                    text,
                    xy=(anchor, pos),
                    xytext=(pad, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontfamily=cfg.font_family,
                    fontsize=cfg.tick_label_fontsize,
                )
            else:
                axis.annotate(
                    text,
                    xy=(pos, anchor),
                    xytext=(0, pad),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontfamily=cfg.font_family,
                    fontsize=cfg.tick_label_fontsize,
                )


def _draw_dot_series(
    axis: Any,
    labels: Sequence[str],
    values: NDArray[np.float64],
    errors: NDArray[np.float64] | None,
    *,
    horizontal: bool,
    annotate: bool,
    cfg: PlottingStyleConfig,
) -> None:
    """Points with optional error bars for values with no meaningful zero baseline."""
    from matplotlib.ticker import ScalarFormatter

    sty = series_style(0, cfg)
    positions = np.arange(len(labels), dtype=np.float64)
    keep = np.isfinite(values)
    err = None
    if errors is not None:
        err = np.where(np.isfinite(errors), errors, 0.0)[keep]
    if horizontal:
        axis.errorbar(
            values[keep], positions[keep], xerr=err, fmt=sty["marker"],
            color=sty["color"], markersize=sty["markersize"] * 1.5,
            elinewidth=sty["linewidth"], capsize=4.0, linestyle="none",
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [str(x) for x in labels],
            fontfamily=cfg.font_family, fontsize=cfg.tick_label_fontsize,
        )
        axis.set_ylim(len(labels) - 0.5, -0.5)
        value_axis = axis.xaxis
    else:
        axis.errorbar(
            positions[keep], values[keep], yerr=err, fmt=sty["marker"],
            color=sty["color"], markersize=sty["markersize"] * 1.5,
            elinewidth=sty["linewidth"], capsize=4.0, linestyle="none",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [str(x) for x in labels],
            fontfamily=cfg.font_family, fontsize=cfg.tick_label_fontsize,
        )
        axis.set_xlim(-0.5, len(labels) - 0.5)
        value_axis = axis.yaxis
    # Large-magnitude values (logZ ~ -1e4) must not collapse into an offset label.
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    value_axis.set_major_formatter(formatter)
    if annotate:
        for pos, value, ok, e in zip(
            positions, values, keep,
            errors if errors is not None else np.full(values.shape, np.nan),
        ):
            if not ok:
                continue
            # Fixed-point text: a point estimate must read off without decoding
            # scientific notation.
            text = f"{float(value):.2f}"
            if np.isfinite(e) and e > 0.0:
                text += f" $\\pm$ {float(e):.2f}"
            xy = (float(value), float(pos)) if horizontal else (float(pos), float(value))
            # Beside the marker, clear of its error bar.
            axis.annotate(
                text, xy=xy, xytext=(0, 12) if horizontal else (12, 0),
                textcoords="offset points", ha="center" if horizontal else "left",
                va="bottom" if horizontal else "center", fontfamily=cfg.font_family,
                fontsize=cfg.tick_label_fontsize,
            )
    if not horizontal:
        axis.margins(y=0.25)
    else:
        axis.margins(x=0.25)


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
    horizontal: bool | Literal["auto"] = "auto",
    log_values: bool | Literal["auto"] = "auto",
    annotate: bool = True,
    baseline_zero: bool = True,
    errors: Sequence[float] | None = None,
) -> Path | None:
    """Bar chart for funnel steps, fit-tier coverage, gate outcomes, rankings.

    Parameters
    ----------
    labels / values:
        One category label and value per bar, drawn in the given order (first
        at the top of a horizontal chart). NaN draws no bar, annotated ``n/a``.
    xlabel / ylabel:
        **Category**-axis and **value**-axis labels. On a horizontal chart the
        category axis is vertical, so ``xlabel`` is drawn on the y axis.
    horizontal:
        ``"auto"`` (default) uses ``bars_should_be_horizontal``: horizontal bars
        when labels would overlap (#333). ``True`` / ``False`` force it.
    log_values:
        ``"auto"`` uses ``choose_bar_value_scale``; ``True`` forces log.
    annotate:
        Print each value beside its bar at tick-label size (a log axis makes
        small counts hard to read off).
    baseline_zero:
        ``False`` draws points (with ``errors`` as error bars) instead of bars,
        for quantities whose zero is meaningless — a bar from 0 to
        ``logZ ≈ -1.8e4`` carries no information (#333). The value axis then
        auto-scales around the data and never uses an offset label.
    errors:
        Optional 1σ uncertainties, used only when ``baseline_zero`` is False.
    """
    if len(labels) != len(values) or not labels:
        return None
    cfg = resolve_plotting_style(style)
    face = color if color is not None else cfg.hist_face_color
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    is_horizontal = (
        bars_should_be_horizontal(labels, cfg) if horizontal == "auto" else bool(horizontal)
    )
    plt = require_pyplot()
    fig, axis = plt.subplots(
        figsize=_bar_figure_size(labels, 1, is_horizontal, cfg)
    )
    if baseline_zero:
        if log_values == "auto":
            scale = choose_bar_value_scale(vals, cfg)
        else:
            scale = "log" if log_values else "linear"
        _draw_bar_series(
            fig, axis, labels, [(None, vals, face, "")],
            horizontal=is_horizontal, scale=scale, annotate=annotate, cfg=cfg,
        )
    else:
        errs = (
            np.asarray([float(e) for e in errors], dtype=np.float64)
            if errors is not None and len(errors) == len(values)
            else None
        )
        _draw_dot_series(
            axis, labels, vals, errs,
            horizontal=is_horizontal, annotate=annotate, cfg=cfg,
        )
    cat_label, value_label = (ylabel, xlabel) if is_horizontal else (xlabel, ylabel)
    apply_axes_style(axis, cfg, xlabel=cat_label, ylabel=value_label, title=title)
    _clear_category_minor_ticks(axis, is_horizontal)
    return save_figure(fig, path, dpi=dpi)


def _clear_category_minor_ticks(axis: Any, horizontal: bool) -> None:
    """Tidy the categorical axis: no minor ticks (they fall between bars) and no
    mirrored ticks on the far side, where value annotations sit."""
    from matplotlib.ticker import NullLocator

    (axis.yaxis if horizontal else axis.xaxis).set_minor_locator(NullLocator())
    if horizontal:
        axis.tick_params(axis="y", which="both", right=False)
    else:
        axis.tick_params(axis="x", which="both", top=False)


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
    horizontal: bool | Literal["auto"] = "auto",
    log_values: bool | Literal["auto"] = "auto",
    annotate: bool = True,
) -> Path | None:
    """Grouped bar chart (mock vs real fractions, attrition passed / failed / N/A).

    Same layout rules as ``plot_categorical_bars``: horizontal bars when labels
    would overlap, log value axis when heights span many decades, every bar
    annotated, and ``xlabel`` / ``ylabel`` are the category / value labels.
    On a horizontal chart the legend sits outside the axes, to the right, so it
    never covers bars (#333).
    """
    if not labels or not series:
        return None
    for values in series.values():
        if len(values) != len(labels):
            return None
    cfg = resolve_plotting_style(style)
    arrays = [
        (str(name), np.asarray([float(v) for v in vals], dtype=np.float64))
        for name, vals in series.items()
    ]
    is_horizontal = (
        bars_should_be_horizontal(labels, cfg) if horizontal == "auto" else bool(horizontal)
    )
    pooled = np.concatenate([a for _n, a in arrays])
    if log_values == "auto":
        scale = choose_bar_value_scale(pooled, cfg)
    else:
        scale = "log" if log_values else "linear"
    plt = require_pyplot()
    fig, axis = plt.subplots(
        figsize=_bar_figure_size(
            labels,
            len(arrays),
            is_horizontal,
            cfg,
            legend_inches=_outside_legend_inches(
                [n for n, _a in arrays], is_horizontal, cfg
            ),
        )
    )
    hatches = list(cfg.bar_hatch_cycle) or [""]
    drawn = [
        (name, arr, series_style(index, cfg)["color"], hatches[index % len(hatches)])
        for index, (name, arr) in enumerate(arrays)
    ]
    _draw_bar_series(
        fig, axis, labels, drawn,
        horizontal=is_horizontal, scale=scale, annotate=annotate, cfg=cfg,
    )
    cat_label, value_label = (ylabel, xlabel) if is_horizontal else (xlabel, ylabel)
    apply_axes_style(axis, cfg, xlabel=cat_label, ylabel=value_label, title=title)
    _clear_category_minor_ticks(axis, is_horizontal)
    if is_horizontal:
        axis.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
            prop=legend_prop(cfg),
        )
    else:
        axis.legend(loc="best", prop=legend_prop(cfg))
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
    log_x: bool | Literal["auto"] = False,
    style: PlottingStyleConfig | None = None,
) -> Path | None:
    """Line plot with optional horizontal threshold (MC/Poisson convergence).

    ``log_x="auto"`` uses a log x axis when the finite, strictly positive x grid
    spans ``plotting.auto_log_min_decades`` or more (#333). Non-finite points
    are not drawn.
    """
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size == 0 or yy.size == 0 or xx.size != yy.size:
        return None
    cfg = resolve_plotting_style(style)
    sty = series_style(0, cfg)
    keep = np.isfinite(xx) & np.isfinite(yy)
    if not np.any(keep):
        return None
    xx, yy = xx[keep], yy[keep]
    if log_x == "auto":
        use_log_x = (
            choose_axis_scale(xx, min_decades=float(cfg.auto_log_min_decades)) == "log"
        )
    else:
        use_log_x = bool(log_x)
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
        axis.legend(loc="best", prop=legend_prop(cfg))
    if use_log_x:
        axis.set_xscale("log")
    if yy.size > 1 and float(np.ptp(yy)) == 0.0:
        # A flat line in an auto-scaled frame looks like structure; say what it is.
        axis.text(
            0.5,
            0.9,
            f"all {yy.size} points equal {float(yy[0]):g}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontfamily=cfg.font_family,
            fontsize=cfg.tick_label_fontsize,
        )
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


#: Inches an outside legend needs beyond its widest label (handle + padding).
_LEGEND_HANDLE_INCHES: Final[float] = 1.1

#: Side gutter (inches) around a reserved caption band.
_CAPTION_GUTTER_INCHES: Final[float] = 0.18


def _caption_band(
    cfg: PlottingStyleConfig, width: float, caption: str | None
) -> tuple[list[str], float]:
    """Wrap ``caption`` to the figure width and return ``(lines, band_inches)``.

    The band is the extra figure height reserved beneath the axes; ``0.0`` and
    no lines when there is no caption. Caption text renders at
    ``tick_label_fontsize`` so nothing on the figure is smaller than it.
    """
    if not caption:
        return [], 0.0
    gutter = _CAPTION_GUTTER_INCHES
    lines = wrap_text_to_inches(
        caption,
        max_width_inches=width - 2.0 * gutter,
        font_family=cfg.font_family,
        fontsize=cfg.tick_label_fontsize,
    )
    line_height = float(cfg.tick_label_fontsize) * float(cfg.caption_line_spacing) / 72.0
    return lines, line_height * len(lines) + 2.0 * gutter


def _save_with_caption_band(
    fig: Any,
    path: Path,
    *,
    dpi: int,
    cfg: PlottingStyleConfig,
    caption_lines: Sequence[str],
    band: float,
) -> Path:
    """Lay out above a reserved caption band, draw the caption, save and close.

    The band is reserved explicitly rather than left to ``tight_layout``, which
    would let long caption text overlap the x-axis label (``docs/PLOTS.md``).
    """
    import warnings

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = (float(v) for v in fig.get_size_inches())
    gutter = _CAPTION_GUTTER_INCHES
    _fit_titles(fig)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0.0, band / height, 1.0, 1.0))
        if _fit_titles(fig):
            fig.tight_layout(rect=(0.0, band / height, 1.0, 1.0))
    fig.text(
        gutter / width,
        (band - gutter) / height,
        "\n".join(caption_lines),
        ha="left",
        va="top",
        fontfamily=cfg.font_family,
        fontsize=cfg.tick_label_fontsize,
        linespacing=float(cfg.caption_line_spacing),
        math_fontfamily=math_fontfamily(cfg),
    )
    bbox = "tight" if figure_overflows(fig) else None
    fig.savefig(path, dpi=dpi, bbox_inches=bbox)
    plt = require_pyplot()
    plt.close(fig)
    return path


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
        ``{"M_Ch": 1.4, "M_TOV": 2.2}``. Each line takes the next
        ``reference_line_style`` (colour and linestyle both differ), so ``M_Ch``
        and ``M_TOV`` are never drawn alike (#333).
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

    caption_lines, band = _caption_band(cfg, width, caption)
    height += band

    # Room for the outside legend: its widest entry plus handle and padding.
    legend_labels = [label for label, _xs, _ys in series] + [
        f"{name}={value:g}" + r" M$_{\odot}$"
        for name, value in (vlines or {}).items()
        if np.isfinite(value)
    ]
    width += _LEGEND_HANDLE_INCHES + max(
        measure_text_width_inches(
            text, font_family=cfg.font_family, fontsize=cfg.legend_fontsize
        )
        for text in legend_labels
    )

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
    for ref_index, (name, value) in enumerate((vlines or {}).items()):
        if not np.isfinite(value):
            continue
        axis.axvline(
            float(value),
            **reference_line_style(ref_index, cfg),
            label=f"{name}={value:g}" + r" M$_{\odot}$",
        )
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title_text)
    # Outside the axes, right: steep curves (a soft M_TOV cut-off) pass through
    # any in-frame corner, and a legend must not cover the signal (#333).
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        prop=legend_prop(cfg),
    )

    if not caption_lines:
        return save_figure(fig, path, dpi=dpi)
    return _save_with_caption_band(
        fig, path, dpi=dpi, cfg=cfg, caption_lines=caption_lines, band=band
    )


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
    axis.legend(loc="best", prop=legend_prop(cfg))
    apply_axes_style(axis, cfg, xlabel=xlabel, ylabel=ylabel, title=title)
    return save_figure(fig, path, dpi=dpi)
