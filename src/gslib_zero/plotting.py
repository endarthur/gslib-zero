"""
Variogram plotting utilities.

- plot_experimental: Plot experimental variogram points
- plot_model: Plot variogram model curve
- plot_variogram: Combined experimental + model overlay
- export_variogram_par: Export model to GSLIB par file format
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from gslib_zero.utils import VariogramModel, evaluate_variogram

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from gslib_zero.variogram import VariogramResult


def plot_experimental(
    results: list[VariogramResult] | VariogramResult,
    ax: Axes | None = None,
    show_pairs: bool = False,
    colors: list[str] | None = None,
    markers: list[str] | None = None,
    labels: list[str] | None = None,
    **kwargs,
) -> Axes:
    """
    Plot experimental variogram points.

    Args:
        results: VariogramResult or list of VariogramResult from gamv
        ax: Matplotlib axes to plot on. If None, creates new figure.
        show_pairs: If True, scale marker size by number of pairs
        colors: List of colors for each variogram direction
        markers: List of marker styles for each direction
        labels: List of labels for legend
        **kwargs: Additional arguments passed to plt.scatter

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]

    # Default colors and markers
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    default_markers = ["o", "s", "^", "D", "v"]

    if colors is None:
        colors = default_colors
    if markers is None:
        markers = default_markers

    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Build label
        if labels is not None and i < len(labels):
            label = labels[i]
        else:
            azm, dip = result.direction
            label = f"Az={azm:.0f}°, Dip={dip:.0f}°"

        # Filter out zero-pair lags
        valid = result.num_pairs > 0
        lag_dist = result.lag_distances[valid]
        gamma = result.gamma[valid]
        pairs = result.num_pairs[valid]

        if show_pairs:
            # Scale marker size by log of pair count
            sizes = 20 + 30 * np.log10(pairs + 1)
            ax.scatter(
                lag_dist, gamma,
                s=sizes, c=color, marker=marker, label=label,
                edgecolors="white", linewidths=0.5,
                **kwargs
            )
        else:
            ax.scatter(
                lag_dist, gamma,
                s=50, c=color, marker=marker, label=label,
                edgecolors="white", linewidths=0.5,
                **kwargs
            )

    ax.set_xlabel("Lag Distance")
    ax.set_ylabel("Gamma (Semivariance)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if len(results) > 1 or labels is not None:
        ax.legend()

    return ax


def plot_model(
    model: VariogramModel,
    max_distance: float,
    n_points: int = 100,
    ax: Axes | None = None,
    color: str = "black",
    linestyle: str = "-",
    linewidth: float = 2.0,
    label: str | None = "Model",
    show_nugget: bool = True,
    show_sill: bool = True,
    **kwargs,
) -> Axes:
    """
    Plot variogram model curve.

    Args:
        model: VariogramModel to plot
        max_distance: Maximum distance to plot to
        n_points: Number of points for smooth curve
        ax: Matplotlib axes. If None, creates new figure.
        color: Line color
        linestyle: Line style
        linewidth: Line width
        label: Label for legend
        show_nugget: If True, show horizontal line at nugget
        show_sill: If True, show horizontal line at total sill
        **kwargs: Additional arguments passed to plt.plot

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Evaluate model
    distances = np.linspace(0, max_distance, n_points)
    gamma = evaluate_variogram(model, distances)

    # Plot model curve
    ax.plot(
        distances, gamma,
        color=color, linestyle=linestyle, linewidth=linewidth,
        label=label, **kwargs
    )

    # Show nugget line
    if show_nugget and model.nugget > 0:
        ax.axhline(
            y=model.nugget, color="gray", linestyle=":",
            linewidth=1, alpha=0.7, label=f"Nugget = {model.nugget:.3f}"
        )

    # Show sill line
    if show_sill:
        total_sill = model.total_sill
        ax.axhline(
            y=total_sill, color="gray", linestyle="--",
            linewidth=1, alpha=0.7, label=f"Sill = {total_sill:.3f}"
        )

    ax.set_xlabel("Lag Distance")
    ax.set_ylabel("Gamma (Semivariance)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    return ax


def plot_variogram(
    experimental: list[VariogramResult] | VariogramResult | None = None,
    model: VariogramModel | None = None,
    max_distance: float | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    show_pairs: bool = False,
    show_nugget: bool = True,
    show_sill: bool = True,
    exp_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
) -> Axes:
    """
    Combined plot of experimental variogram + model overlay.

    Args:
        experimental: VariogramResult(s) from gamv
        model: VariogramModel to overlay
        max_distance: Maximum distance for model curve. If None, uses max
                     experimental distance or model range.
        ax: Matplotlib axes. If None, creates new figure.
        title: Plot title
        show_pairs: Scale experimental points by pair count
        show_nugget: Show nugget line on model plot
        show_sill: Show sill line on model plot
        exp_kwargs: Additional kwargs for plot_experimental
        model_kwargs: Additional kwargs for plot_model

    Returns:
        Matplotlib Axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if exp_kwargs is None:
        exp_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}

    # Determine max distance
    if max_distance is None:
        if experimental is not None:
            if isinstance(experimental, list):
                max_distance = max(r.lag_distances.max() for r in experimental)
            else:
                max_distance = experimental.lag_distances.max()
        elif model is not None and model.structures:
            # Use 1.5x the major range
            max_distance = 1.5 * max(s["ranges"][0] for s in model.structures)
        else:
            max_distance = 100.0

    # Plot model first (so experimental points are on top)
    if model is not None:
        plot_model(
            model, max_distance,
            ax=ax, show_nugget=show_nugget, show_sill=show_sill,
            **model_kwargs
        )

    # Plot experimental
    if experimental is not None:
        plot_experimental(
            experimental, ax=ax, show_pairs=show_pairs,
            **exp_kwargs
        )

    if title:
        ax.set_title(title)

    ax.legend()

    return ax


def export_variogram_par(
    model: VariogramModel,
    filepath: Path | str,
    comment_style: str = "!",
) -> None:
    """
    Export variogram model to GSLIB par file snippet.

    Output format matches GSLIB variogram specification:
    ```
    nst nugget
    type sill ang1 ang2 ang3
    range1 range2 range3
    ```

    Args:
        model: VariogramModel to export
        filepath: Output file path
        comment_style: Comment character ('!' for GSLIB, '#' for Python)
    """
    filepath = Path(filepath)

    lines = []
    lines.append(f"{comment_style} Variogram model exported from gslib-zero")
    lines.append(f"{len(model.structures)} {model.nugget}  {comment_style} nst, nugget")

    for i, struct in enumerate(model.structures):
        vtype = int(struct["type"])
        sill = struct["sill"]
        angles = struct.get("angles", (0.0, 0.0, 0.0))
        ranges = struct["ranges"]

        lines.append(
            f"{vtype} {sill} {angles[0]} {angles[1]} {angles[2]}  "
            f"{comment_style} structure {i+1}: type, sill, angles"
        )
        lines.append(
            f"{ranges[0]} {ranges[1]} {ranges[2]}  "
            f"{comment_style} ranges (major, minor, vertical)"
        )

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
