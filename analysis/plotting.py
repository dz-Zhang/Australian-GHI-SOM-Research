"""Plotting helpers for GHI DataArrays and plotting SOM results."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.collections import QuadMesh
import matplotlib.ticker as mticker

AUSTRALIA_EXTENT = (112.0, 156.5, -44.5, -10.0)
GHI_VMIN = -0.5
GHI_VMAX = 43
GHI_CMAP = "RdYlBu_r"

"""GHI Plotting helpers"""
class GHIPlotter:
    """Plotting interface for gridded irradiance data."""

    EXTENT = AUSTRALIA_EXTENT

    @staticmethod
    def plot_ghi(
        da: xr.DataArray,
        extent: tuple[float, float, float, float] | None = AUSTRALIA_EXTENT,
        cmap=GHI_CMAP,
        ax=None,
        **plot_kwargs,
    ) -> QuadMesh:
        return plot_ghi(da, extent=extent, cmap=cmap, ax=ax, **plot_kwargs)

    @staticmethod
    def format_labels(da: xr.DataArray, ax, **kwargs) -> None:
        format_labels(da, ax, **kwargs)


def plot_ghi(da: xr.DataArray, extent=AUSTRALIA_EXTENT, cmap=GHI_CMAP, ax=None, **plot_kwargs) -> QuadMesh:
    """Thin wrapper around xarray's built-in plot on a Cartopy map axes."""
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
    title = plot_kwargs.pop("title", None)
    add_colorbar = plot_kwargs.pop("add_colorbar", False)
    plot_kwargs.setdefault("vmin", GHI_VMIN)
    plot_kwargs.setdefault("vmax", GHI_VMAX)
    mesh = ax.pcolormesh(da.longitude, da.latitude, da.data, cmap=cmap, **plot_kwargs)
    if add_colorbar:
        ax.figure.colorbar(mesh, ax=ax)
    ax.coastlines()
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title or "Global Horizontal Irradiance")
    values = da.values
    ax.text(0.5, -0.1, f"Min: {np.nanmin(values):.2f}    Max: {np.nanmax(values):.2f}",
            transform=ax.transAxes, ha="center", fontsize=9)
    return mesh

def format_labels(da, ax, **kwargs):
    """Format geographic labels on a Cartopy axis."""
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=1, linestyle="--", **kwargs)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(da.longitude.values)
    gl.ylocator = mticker.FixedLocator(da.latitude.values)
    return gl


"""SOM Plotting helpers"""
def plot_weekly_snapshot_nodes(
    som, hit_map, valid, latitude, longitude, n_samples,
    cmap=GHI_CMAP, units_label="GHI", sample_label="week", title=None,
    node_weights=None,
):
    """Plot one spatial map for each trained SOM node."""
    from .analysis import unflatten_to_grid

    rows, cols = som.x, som.y
    weights = (som.weights.detach().cpu().numpy()
               if node_weights is None else np.asarray(node_weights))
    node_maps = [
        [unflatten_to_grid(weights[r, c], valid, latitude, longitude) for c in range(cols)]
        for r in range(rows)
    ]
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 2.8 * rows),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
    )
    mesh = None
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            mesh = GHIPlotter.plot_ghi(
                node_maps[r][c], ax=ax, extent=GHIPlotter.EXTENT, cmap=cmap,
                vmin=GHI_VMIN, vmax=GHI_VMAX, add_colorbar=False,
            )
            for txt in list(ax.texts):
                txt.remove()
            hits = int(hit_map[r, c])
            ax.set_title(
                f"Node ({r},{c}): {hits}/{n_samples} {sample_label}s ({100 * hits / n_samples:.0f}%)",
                fontsize=9,
            )
    fig.suptitle(title or f"Weekly-mean SOM nodes ({rows}x{cols})", fontsize=13)
    fig.subplots_adjust(right=0.9, wspace=0.15, hspace=0.4)
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    fig.colorbar(mesh, cax=cbar_ax, label=units_label)
    return fig, axes

def plot_weekly_sequence_nodes(som, hit_map, valid, latitude, longitude, n_weeks, cmap=GHI_CMAP, units_label="GHI"):
    """Plot each SOM node as a sequence of daily spatial maps."""
    from .analysis import unflatten_to_grid

    rows, cols = som.x, som.y
    weights = som.weights.detach().cpu().numpy()
    n_valid = int(valid.sum())
    sequence_len = weights.shape[-1] // n_valid
    node_cubes = weights.reshape(rows, cols, sequence_len, n_valid)
    lon_min, lon_max, lat_min, lat_max = GHIPlotter.EXTENT
    map_aspect = (lon_max - lon_min) / (lat_max - lat_min)
    per_map_h = 1.1
    fig = plt.figure(figsize=(per_map_h * map_aspect * cols * sequence_len + 1.5, per_map_h * rows + 1.0))
    outer = fig.add_gridspec(rows, cols, wspace=0.35, hspace=0.5)
    mesh = None
    for r in range(rows):
        for c in range(cols):
            inner = outer[r, c].subgridspec(1, sequence_len, wspace=0.03)
            for d in range(sequence_len):
                ax = fig.add_subplot(inner[0, d], projection=ccrs.PlateCarree())
                day_map = unflatten_to_grid(node_cubes[r, c, d], valid, latitude, longitude)
                mesh = GHIPlotter.plot_ghi(
                    day_map, ax=ax, extent=GHIPlotter.EXTENT, cmap=cmap,
                    vmin=GHI_VMIN, vmax=GHI_VMAX, add_colorbar=False,
                )
                for txt in list(ax.texts):
                    txt.remove()
                ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
                if d == 0:
                    hits = int(hit_map[r, c])
                    ax.text(-0.35, 0.5, f"({r},{c})\n{hits}/{n_weeks}", transform=ax.transAxes,
                            ha="right", va="center", fontsize=7)
    fig.suptitle(f"Part 2: {sequence_len}-day-sequence SOM nodes ({rows}x{cols}, sequence steps left to right)", fontsize=13)
    fig.colorbar(mesh, ax=fig.axes, shrink=0.6, pad=0.02, label=units_label)
    return fig

def compare_soms(snapshot_som, sequence_som):
    """Compare snapshot SOM nodes with day-averaged sequence SOM nodes."""
    rows, cols = snapshot_som.x, snapshot_som.y
    if (sequence_som.x, sequence_som.y) != (rows, cols):
        raise ValueError(f"compare_soms: grid shapes differ ({rows}x{cols} vs {sequence_som.x}x{sequence_som.y})")
    w1 = snapshot_som.weights.detach().cpu().numpy()
    w2 = sequence_som.weights.detach().cpu().numpy()
    n_valid = w1.shape[-1]
    sequence_len = w2.shape[-1] // n_valid
    w2_day_avg = w2.reshape(rows, cols, sequence_len, n_valid).mean(axis=2)
    correlations = np.array([[np.corrcoef(w1[r, c], w2_day_avg[r, c])[0, 1] for c in range(cols)] for r in range(rows)])
    print("compare_soms: per-node Pearson correlation, part-1 node vs day-averaged part-2 node:")
    print(np.round(correlations, 3))
    print(f"compare_soms: mean correlation {correlations.mean():.3f}, min {correlations.min():.3f}")
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    im = ax.imshow(correlations, cmap="RdYlGn", vmin=-1, vmax=1)
    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, f"{correlations[r, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    ax.set_xlabel("node col"); ax.set_ylabel("node row")
    ax.set_title("Part 1 vs day-averaged Part 2\n(Pearson r per node)")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    return fig, correlations
