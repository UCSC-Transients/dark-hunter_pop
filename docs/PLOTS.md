# Plotting style guide

Authoritative visual conventions for diagnostic and paper-ready figures in
`dark-hunter_pop`. Shared primitives live in `src/darkhunter_pop/plotting.py`.
Numeric style defaults live under **`plotting:`** in `config/config.yaml` (draft:
`config/fragments/plotting.yaml`). Layout / DPI / hook enable flags stay under
`diagnostics:`.

Plot captions and diagnostic reports are full-detail (caveman exemption).

## Config ownership

| Concern | Config block | Notes |
| --- | --- | --- |
| Fonts, ticks, line weights, color/linestyle/marker cycles | `plotting` | Display-only; not stage checksum input |
| `figure_dpi`, write flags, hook toggles | `diagnostics` | Figure output layout |
| Histogram bin cap (`histogram_max_bins`) | `diagnostics` | Avoids empty heavy-tailed histos (#96) |
| Sky-map marker size / alpha | `diagnostics` | NSS-scale Mollweide (#97) |

Load style with `load_config().plotting` and pass `style=` into plotting helpers, or
call `apply_axes_style(ax, style)` / `series_style(i, style)` from custom figures.

## Color and series discrimination

All series must remain distinguishable by **shape, linestyle, and color** together:

- Use the Okabe–Ito palette from [Color Universal Design](https://jfly.uni-koeln.de/color/)
  (`plotting.color_cycle`). Do not invent ad-hoc rainbows or red/green-only pairs.
- Cycle **linestyles** and **markers** with color (`plotting.linestyle_cycle`,
  `plotting.marker_cycle`) so plots stay readable for color-vision deficiency and
  when printed black-and-white.
- Prefer `series_style(index, style)` rather than hardcoding hex codes in call sites.

## Axes, ticks, and weight

Defaults match this pattern (values from `plotting:`):

```python
ax.set_xlabel(r"Right Ascension (deg)", fontfamily="serif", fontsize=18)
ax.set_ylabel(r"Declination (deg)", fontfamily="serif", fontsize=18)
ax.tick_params(
    axis="both",
    which="both",
    right=True,
    top=True,
    width=2,
    length=8,
    direction="in",
    labelsize=14,
)
ax.tick_params(axis="x", which="minor", length=4, width=2, direction="in")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
```

Rules:

- **Serif** fonts (`plotting.font_family`).
- **Inward** ticks on **all four sides**; enable **minor** ticks wherever the
  projection allows (`apply_axes_style`).
- Spines, ticks, and data lines are **relatively thick** (`spines_width`,
  `tick_width`, `line_width`). Thin hairlines are not acceptable for product figures.
- Prefer `apply_axes_style` after drawing data so labels and ticks stay consistent.

## Aspect ratio and dynamic range

- Choose figsize so the **science relationship** is obvious. Light curves and other
  time series with rise/fall structure are generally **portrait**
  (`plotting.figsize_portrait`). Sky maps and wide funnel bars use
  `figsize_wide` / `figsize_landscape`.
- Set axis limits / scales so the relevant dynamic range fills the panel. Do not
  leave the signal as a one-pixel spike in a vast empty frame (see histogram bin
  capping below). Use log axes when the distribution spans many decades **and**
  the science question is multiplicative.

## Histograms: `bins="auto"` and NaNs

- Matplotlib `bins="auto"` (Freedman–Diaconis) on heavy-tailed large-N samples
  (RUWE, orbital period) can produce hundreds–thousands of **sub-pixel** bars that
  look empty. Cap with `diagnostics.histogram_max_bins` via `plot_histogram(...,
  max_bins=...)`.
- Drop **non-finite** values before binning. NaN/inf make `bins="auto"` raise
  `ValueError: autodetected range of [nan, nan] is not finite`. Shared helpers
  already filter finite entries; custom histos must do the same.

## Labels, units, and text

- **Always label axes with units**, e.g. `Right Ascension (deg)`,
  `period (day)`, `mass (M$_\\odot$)`.
- Units use **correct scientific notation without a slash**: write
  `km s$^{-1}$`, not `km/s`; `M_\\odot yr$^{-1}$`, not `Msun/yr`.
- Prefer **prefixed / named units** with O(1)–O(100) tick values: `Gyr` not
  `10^9 year`; `10^{19}\\,{\\rm cm}` as the **unit** with values near unity, not
  raw `1e19` tick labels on `cm`.
- Text must not overlap other text, axes, labels, or critical data. Use
  `tight_layout` / constrained layout, angled tick labels only when needed, and
  legends placed clear of the signal.
- Axis / legend / title text must be **at least as large as caption font size**
  (config defaults: labels 18 pt, ticks/legend 14 pt). Do not shrink annotation
  text below the tick size to “make it fit.”

## Checklist before merging a figure change

1. Colorblind- and B/W-safe series discrimination (color + linestyle/marker).
2. Thick lines/ticks/spines; inward major+minor ticks; serif labels with units.
3. Aspect ratio and limits show the intended relationship.
4. Histograms: finite data only; `max_bins` when tails are heavy.
5. No overlapping text; sizes ≥ caption; units without `/`; sensible unit prefixes.
6. Style knobs read from `plotting:` / `diagnostics:` — no new magic numbers in
   call sites.
