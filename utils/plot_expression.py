import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree


def save_expression_plot(fig, output_dir, slide, gene, kind, *, dpi=300):
    """Save a slide/gene plot with the shared naming scheme and close it."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_gene = "".join(c if c.isalnum() or c in "-_" else "_" for c in gene)
        output_path = output_dir / f"{slide}_{safe_gene}_{kind}.png"
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        return output_path
    finally:
        plt.close(fig)


def plot_expression(coords, values, *, title="", colorbar_label="Expression",
                    cmap="viridis", vmin=None, vmax=None, dot_size=2.5,
                    point_spacing=2.4, dpi=300):
    fig, axes = _plot_expression_panels(
        coords, [values], [title], colorbar_label=colorbar_label, cmap=cmap,
        vmin=vmin, vmax=vmax, dot_size=dot_size, point_spacing=point_spacing, dpi=dpi,
    )
    return fig, axes[0]


def plot_expression_comparison(coords, target, prediction, *, title="",
                               colorbar_label="Expression counts", cmap="viridis",
                               vmin=None, vmax=None, dot_size=2.5,
                               point_spacing=2.4, dpi=300):
    """Draw GT and prediction with identical positions and one shared colorbar."""
    prefix = f"{title} · " if title else ""
    return _plot_expression_panels(
        coords, [target, prediction], [f"{prefix}GT", f"{prefix}prediction"],
        colorbar_label=colorbar_label, cmap=cmap, vmin=vmin, vmax=vmax,
        dot_size=dot_size, point_spacing=point_spacing, dpi=dpi,
    )


def _plot_expression_panels(coords, value_sets, titles, *, colorbar_label, cmap,
                            vmin, vmax, dot_size, point_spacing, dpi):
    """Share geometry and normalization between single and paired plots."""

    coords = np.asarray(coords, dtype=float)
    values = np.asarray(value_sets, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        raise ValueError("coords must have shape [N, 2], with N >= 2.")
    if values.shape != (len(titles), len(coords)):
        raise ValueError("Each values array must have shape [N] and match coords row order.")
    if not np.isfinite(coords).all() or not np.isfinite(values).all():
        raise ValueError("coords and values must be finite.")
    if any(not np.isfinite(v) or v <= 0 for v in (dot_size, point_spacing, dpi)):
        raise ValueError("dot_size, point_spacing and dpi must be finite and positive.")

    lower_color = float(min(0, values.min())) if vmin is None else float(vmin)
    upper_color = float(values.max()) if vmax is None else float(vmax)
    if vmax is None and upper_color <= lower_color:
        upper_color = lower_color + 1
    if not np.isfinite([lower_color, upper_color]).all() or upper_color <= lower_color:
        raise ValueError("Color limits must be finite, with vmax > vmin.")
    norm = Normalize(vmin=lower_color, vmax=upper_color)
    # Validate the colormap before creating a figure.
    plt.get_cmap(cmap)

    nearest = cKDTree(coords).query(coords, k=2)[0][:, 1]
    if (nearest <= 0).any():
        raise ValueError("Duplicate coordinates must be resolved before plotting.")
    pitch = float(np.median(nearest))
    if nearest.min() / pitch * point_spacing <= np.sqrt(dot_size):
        raise ValueError("Markers would overlap; increase point_spacing.")
    lower = coords.min(axis=0) - pitch
    upper = coords.max(axis=0) + pitch
    width, height = (upper - lower) * point_spacing / (72 * pitch)
    left, right, bottom, top = 0.18, 1.0, 0.3, 0.5
    panels = len(titles)
    # Leave room for both titles even when a slide occupies a narrow area.
    gap = max(0.45, 2.6 - float(width)) if panels > 1 else 0.45
    panel_width = panels * width + (panels - 1) * gap
    fig_width = max(float(panel_width) + left + right, 3.6 * panels)
    fig_height = max(float(height) + bottom + top, 2.8)
    x0 = left + (fig_width - left - right - panel_width) / 2
    y0 = bottom + (fig_height - bottom - top - height) / 2

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    axes = []
    for index, (column, title) in enumerate(zip(values, titles)):
        panel_x = x0 + index * (width + gap)
        ax = fig.add_axes([panel_x / fig_width, y0 / fig_height,
                           width / fig_width, height / fig_height])
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(upper[1], lower[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, pad=10)
        points = ax.scatter(coords[:, 0], coords[:, 1], c=column, cmap=cmap,
                            norm=norm, s=dot_size, linewidths=0, edgecolors="none")
        axes.append(ax)
    bar_height = min(float(height), 2.4)
    cax = fig.add_axes([(x0 + panel_width + 0.16) / fig_width,
                        (y0 + (height - bar_height) / 2) / fig_height,
                        0.08 / fig_width, bar_height / fig_height])
    colorbar = fig.colorbar(points, cax=cax)
    colorbar.set_label(colorbar_label, fontsize=8)
    colorbar.ax.tick_params(labelsize=8)
    return fig, axes
