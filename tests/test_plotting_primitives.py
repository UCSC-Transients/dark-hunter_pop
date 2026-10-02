"""Unit tests for the readable-by-default plotting primitives (#333, #360)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from darkhunter_pop.plotting import (
    AXIS_LABELS,
    axis_label,
    bars_should_be_horizontal,
    choose_axis_scale,
    choose_bar_value_scale,
    default_plotting_style,
    figure_overflows,
    format_bar_value,
    layout_figure,
    legend_prop,
    matplotlib_available,
    reference_line_style,
    scaled_histogram_edges,
    symlog_major_ticks,
    wrap_label_to_inches,
)

pytestmark = pytest.mark.unit

needs_mpl = pytest.mark.skipif(not matplotlib_available(), reason="matplotlib not installed")


# --- units / label mapping ----------------------------------------------------


@pytest.mark.parametrize("key", ["m2_msun", "period_day", "g_mag", "ra_deg", "dec_deg"])
def test_axis_label_maps_raw_keys_to_labels_with_units(key: str) -> None:
    label = axis_label(key)
    assert label != key
    assert "(" in label and ")" in label, label  # a unit in parentheses
    assert "/" not in label.replace(r"\rm", "")  # PLOTS.md: no slash units


def test_axis_label_f_m_is_a_mass_function_not_a_fraction() -> None:
    assert "fraction" not in AXIS_LABELS["f_m_msun"]
    assert "mass function" in AXIS_LABELS["f_m_msun"]


def test_axis_label_passes_unknown_text_through() -> None:
    assert axis_label("survival of the fittest") == "survival of the fittest"


# --- automatic log / symlog selection ----------------------------------------


def test_choose_axis_scale_log_for_heavy_positive_tail() -> None:
    rng = np.random.default_rng(1)
    # Real #301 RUWE spans 0.54 to 81 (2.2 decades).
    ruwe_like = np.concatenate([rng.normal(1.2, 0.1, 1000).clip(0.6), [0.54, 80.0]])
    assert choose_axis_scale(ruwe_like, min_decades=2.0) == "log"


def test_choose_axis_scale_linear_for_narrow_data() -> None:
    assert choose_axis_scale(np.linspace(8.0, 18.0, 100), min_decades=2.0) == "linear"


def test_choose_axis_scale_linear_for_bounded_data_with_zeros() -> None:
    ecc = np.concatenate([np.zeros(10), np.linspace(0.01, 0.97, 200)])
    assert choose_axis_scale(ecc, min_decades=2.0) == "linear"


def test_choose_axis_scale_symlog_for_signed_heavy_tails() -> None:
    dbic = np.array([-3.6e7, -1.5e5, -300.0, -26.0, -3.0, 0.0, 200.0, 3.9e3, 1.7e7])
    assert choose_axis_scale(dbic, min_decades=2.0) == "symlog"


def test_choose_axis_scale_ignores_nonfinite_and_tiny_inputs() -> None:
    assert choose_axis_scale([np.nan, np.inf, 1.0], min_decades=2.0) == "linear"
    assert choose_axis_scale([], min_decades=2.0) == "linear"


def test_scaled_histogram_edges_are_log_uniform_and_capped() -> None:
    values = np.geomspace(0.2, 1e4, 5000)
    edges = scaled_histogram_edges(values, scale="log", bins="auto", max_bins=40)
    assert edges.size - 1 <= 40
    assert np.allclose(np.diff(np.log10(edges)), np.diff(np.log10(edges))[0])
    assert edges[0] == pytest.approx(0.2) and edges[-1] == pytest.approx(1e4)


def test_symlog_major_ticks_thin_to_the_cap() -> None:
    ticks = symlog_major_ticks(-1e8, 2e7, 26.0, max_ticks=7)
    assert 0.0 in ticks
    assert len(ticks) <= 7
    assert all(t == 0.0 or abs(t) >= 26.0 for t in ticks)


def test_bar_value_scale_logs_counts_spanning_decades() -> None:
    assert choose_bar_value_scale([48, 27, 7741]) == "log"
    assert choose_bar_value_scale([25, 24, 2, 1]) == "linear"
    assert choose_bar_value_scale([-1.0, 5.0, 5000.0]) == "linear"


def test_format_bar_value() -> None:
    assert format_bar_value(7741.0) == "7741"
    assert format_bar_value(0.0) == "0"
    assert format_bar_value(float("nan")) == "n/a"
    assert format_bar_value(0.35214) == "0.3521"


# --- styles -------------------------------------------------------------------


def test_reference_lines_never_share_a_style() -> None:
    first, second = reference_line_style(0), reference_line_style(1)
    assert first["color"] != second["color"]
    assert first["linestyle"] != second["linestyle"]


def test_legend_prop_carries_the_configured_size() -> None:
    cfg = default_plotting_style()
    assert legend_prop(cfg)["size"] == pytest.approx(cfg.legend_fontsize)


@needs_mpl
def test_legend_text_renders_at_legend_fontsize(tmp_path: Path, monkeypatch) -> None:
    """#360: matplotlib ignores fontsize= when prop= is given."""
    from darkhunter_pop import plotting

    cfg = default_plotting_style().model_copy(update={"legend_fontsize": 21.0})
    seen: list[float] = []
    real_save = plotting.save_figure

    def _capture(fig, path, *, dpi):
        for axis in fig.axes:
            legend = axis.get_legend()
            if legend is not None:
                seen.extend(t.get_fontsize() for t in legend.get_texts())
        return real_save(fig, path, dpi=dpi)

    monkeypatch.setattr(plotting, "save_figure", _capture)
    plotting.plot_overlay_histograms(
        {"mock": [1.0, 2.0, 3.0], "real": [1.5, 2.5]},
        tmp_path / "overlay.png",
        xlabel="x",
        title="t",
        dpi=40,
        style=cfg,
    )
    assert seen and all(size == pytest.approx(21.0) for size in seen)


# --- layout: categorical labels and clipping ---------------------------------


@needs_mpl
def test_many_or_long_category_labels_go_horizontal() -> None:
    cfg = default_plotting_style()
    many = [f"cut_{i}" for i in range(cfg.categorical_max_vertical_labels + 1)]
    assert bars_should_be_horizontal(many, cfg)
    assert bars_should_be_horizontal(
        ["astrometric:primary_ns_bh:main_sequence", "spectroscopic:mass_route"], cfg
    )
    assert not bars_should_be_horizontal(["passed", "failed", "skipped"], cfg)


@needs_mpl
def test_wrap_label_keeps_mathtext_whole() -> None:
    text = r"a long title about $M_{\rm Ch}$ and $M_{\rm TOV}$ reference lines " * 3
    wrapped = wrap_label_to_inches(
        text, max_width_inches=2.0, font_family="serif", fontsize=18.0
    )
    assert "\n" in wrapped
    for line in wrapped.splitlines():
        assert line.count("$") % 2 == 0, line


@needs_mpl
def test_layout_wraps_overlong_title_so_nothing_clips() -> None:
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot

    plt = require_pyplot()
    cfg = default_plotting_style()
    fig, axis = plt.subplots(figsize=(4.0, 3.0))
    axis.plot([0, 1], [0, 1])
    apply_axes_style(
        axis,
        cfg,
        xlabel="x",
        ylabel="y",
        title="andrews2022_modified attrition: passed / failed / not applicable " * 2,
    )
    try:
        layout_figure(fig)
        assert "\n" in axis.get_title()
        assert not figure_overflows(fig)
    finally:
        plt.close(fig)


@needs_mpl
def test_grouped_bar_figure_has_no_overflowing_text(tmp_path: Path, monkeypatch) -> None:
    """The El-Badry 2026 waterfall shape: ~20 long cut ids, 3 series, decades of counts."""
    from darkhunter_pop import plotting

    labels = [f"astrometric:sub_chandrasekhar:cut_number_{i}" for i in range(20)]
    series = {
        "passed": [float(10 ** (i % 6)) for i in range(20)],
        "failed": [float(168000 - i) for i in range(20)],
        "not_applicable": [0.0] * 20,
    }
    overflow: list[bool] = []
    real_save = plotting.save_figure

    def _capture(fig, path, *, dpi):
        plotting.layout_figure(fig)
        overflow.append(plotting.figure_overflows(fig))
        assert fig.axes[0].get_xscale() == "log"
        return real_save(fig, path, dpi=dpi)

    monkeypatch.setattr(plotting, "save_figure", _capture)
    out = plotting.plot_grouped_bars(
        labels, series, tmp_path / "waterfall.png",
        xlabel="cut", ylabel="count", title="waterfall", dpi=40,
    )
    assert out is not None and out.exists()
    assert overflow == [False]


@needs_mpl
def test_sky_map_has_axis_labels(tmp_path: Path, monkeypatch) -> None:
    from darkhunter_pop import plotting

    labels: list[tuple[str, str]] = []
    real_save = plotting.save_figure

    def _capture(fig, path, *, dpi):
        axis = fig.axes[0]
        labels.append((axis.get_xlabel(), axis.get_ylabel()))
        return real_save(fig, path, dpi=dpi)

    monkeypatch.setattr(plotting, "save_figure", _capture)
    plotting.plot_sky_mollweide(
        [10.0, 200.0, 300.0], [-30.0, 0.0, 45.0], tmp_path / "sky.png", dpi=40
    )
    assert labels == [(AXIS_LABELS["ra_deg"], AXIS_LABELS["dec_deg"])]


@needs_mpl
def test_histogram_auto_log_axis_for_heavy_tail(tmp_path: Path, monkeypatch) -> None:
    from darkhunter_pop import plotting

    scales: list[str] = []
    real_save = plotting.save_figure

    def _capture(fig, path, *, dpi):
        scales.append(fig.axes[0].get_xscale())
        return real_save(fig, path, dpi=dpi)

    monkeypatch.setattr(plotting, "save_figure", _capture)
    period = np.geomspace(0.2, 1e4, 2000)
    plotting.plot_histogram(period, tmp_path / "p.png", xlabel="P", title="t", dpi=40, max_bins=50)
    plotting.plot_histogram(
        np.linspace(0, 0.9, 200), tmp_path / "e.png", xlabel="e", title="t", dpi=40
    )
    plotting.plot_histogram(
        [-3e7, -2e3, -30.0, -5.0, 1.0, 400.0, 2e7],
        tmp_path / "d.png", xlabel="d", title="t", dpi=40,
    )
    assert scales == ["log", "linear", "symlog"]


@needs_mpl
def test_dndm_reference_lines_are_styled_apart(tmp_path: Path, monkeypatch) -> None:
    from darkhunter_pop import plotting

    styles: list[tuple[str, str]] = []
    real_save = plotting.save_figure

    def _capture(fig, path, *, dpi):
        for line in fig.axes[0].get_lines():
            if line.get_label().startswith("M_"):
                styles.append((str(line.get_color()), str(line.get_linestyle())))
        return real_save(fig, path, dpi=dpi)

    monkeypatch.setattr(plotting, "save_figure", _capture)
    grid = np.geomspace(0.2, 20.0, 30)
    plotting.plot_dndm_by_class(
        grid, np.ones_like(grid), {"WD": np.ones_like(grid)}, tmp_path / "d.png",
        dpi=40, class_order=["WD"], vlines={"M_Ch": 1.4, "M_TOV": 2.2},
    )
    assert len(styles) == 2 and styles[0] != styles[1]
    assert styles[0][0] != styles[1][0] and styles[0][1] != styles[1][1]
