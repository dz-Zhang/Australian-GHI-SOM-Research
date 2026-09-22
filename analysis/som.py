from pathlib import Path
from typing import Any, Literal
from dataclasses import asdict
import warnings
import json
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torchsom import SOM as TorchSOM, SOMVisualizer
from tqdm import tqdm
from torchsom.visualization import VisualizationConfig

from .analysis import (
    AnalysisBase,
    DataConfig,
)
from .dataloader import DataLoader
from .plotting import (
    GHIPlotter,
    GHI_CMAP,
    GHI_VMIN,
    GHI_VMAX,
    plot_weekly_sequence_nodes,
    plot_weekly_snapshot_nodes,
)
from .dim_reduce_clustering import PCAClustering

# Define default training arguments
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 4
SIGMA = 2
LEARNING_RATE = 0.5
RANDOM_SEED = 0
DEFAULT_TRAINING_ARGS = dict(
        topology="hexagonal",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        random_seed=RANDOM_SEED,
        initialization_mode="random",
        lr_decay_function="lr_linear_decay_to_zero",
        sigma_decay_function="sig_linear_decay_to_one",
        sigma=SIGMA,
        learning_rate=LEARNING_RATE
        )


class _SOMCore(AnalysisBase):
    """Self-organising map analysis using the shared preparation pipeline."""

    def __init__(self, data: Any, config: DataConfig | None = None) -> None:
        super().__init__(data, config)
        self.model: TorchSOM | None = None
        self.training_metrics: dict[str, Any] = {}
        self.hit_map: np.ndarray | None = None
        self.bmus: np.ndarray | None = None
        self.pca_model = None
        self.training_feature_matrix: np.ndarray | None = None
        self.pca_n_components: int | None = None

    def fit(
        self,
        rows: int,
        cols: int,
        *,
        verbose: bool = True,
        plot_errors: bool = False,
        output_dir = None,
        **som_kwargs: Any,
    ) -> "SOM":
        """Train the SOM, optionally after PCA reduction from DataConfig."""
        raw_X = self.feature_matrix if self.feature_matrix is not None else self.prepare()
        X = raw_X
        self.pca_model = None
        self.pca_n_components = None
        if self.config.pca:
            pca_analysis = PCAClustering(self.data, self.config)
            threshold = 1.0 if self.config.pca_threshold is None else self.config.pca_threshold
            X, self.pca_model, self.pca_n_components = pca_analysis.reduce_dimensions(
                variance_threshold=threshold,
                max_components=self.config.pca_max_ncomp,
            )
        self.training_feature_matrix = np.asarray(X, dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model, _, self.hit_map, config = train_som(
                X, rows, cols, verbose=verbose, plot_errors=plot_errors, output_dir=output_dir, **som_kwargs
            )
        self._record_warnings(caught)
        self.bmus = None
        data = torch.tensor(X, dtype=torch.float32)
        self.training_metrics = {
            "config": config,
            "quantization_error": float(self.model.quantization_error(data)),
            "topographic_error": float(self.model.topographic_error(data)),
        }
        if self.pca_model is not None:
            self.training_metrics["pca"] = {
                "enabled": True,
                "threshold": self.config.pca_threshold,
                "max_ncomp": self.config.pca_max_ncomp,
                "n_components": self.pca_n_components,
                "explained_variance": float(
                    self.pca_model.explained_variance_ratio_[:self.pca_n_components].sum()
                ),
            }
        return self

    def _training_matrix(self) -> np.ndarray:
        if self.training_feature_matrix is not None:
            return self.training_feature_matrix
        if self.feature_matrix is None:
            raise RuntimeError("fit() must be called before accessing training data")
        return np.asarray(self.feature_matrix, dtype=float)

    def node_weights_in_original_space(self) -> np.ndarray:
        """Return learned node weights reconstructed to the prepared feature space."""
        if self.model is None:
            raise RuntimeError("fit() must be called before accessing node weights")
        weights = self.model.weights.detach().cpu().numpy()
        if self.pca_model is None:
            return weights
        rows, cols, n_components = weights.shape
        full_components = np.zeros(
            (rows * cols, self.pca_model.n_components_), dtype=weights.dtype
        )
        full_components[:, :n_components] = weights.reshape(-1, n_components)
        return self.pca_model.inverse_transform(
            full_components
        ).reshape(rows, cols, -1)

    def average_bmu_correlations(self) -> np.ndarray:
        """Calculate the average Pearson correlation between each SOM node
        and the prepared samples assigned to that node as their BMU.

        Returns a ``(rows, cols)`` array indexed in the same ``(row, col)``
        convention as ``hit_map`` and ``identify_bmus``. Nodes without any
        assigned samples, or correlations that cannot be calculated because
        one vector is constant, are represented by ``numpy.nan``.
        """
        if self.model is None or self.training_feature_matrix is None:
            raise RuntimeError("fit() must be called before calculating BMU correlations")

        data = self._training_matrix()
        weights = self.model.weights.detach().cpu().numpy()
        bmus = self._ensure_bmus()
        rows, cols = int(self.model.x), int(self.model.y)
        correlations = np.full((rows, cols), np.nan, dtype=float)

        for row in range(rows):
            for col in range(cols):
                assigned = data[
                    (bmus[:, 0] == row) & (bmus[:, 1] == col)
                ]
                if assigned.shape[0] == 0:
                    continue

                node = np.asarray(weights[row, col], dtype=float)
                node_std = np.std(node)
                sample_std = np.std(assigned, axis=1)
                valid = (node_std > 0) & (sample_std > 0)
                if not np.any(valid):
                    continue

                sample_correlations = np.corrcoef(
                    np.vstack((assigned[valid], node))
                )[:-1, -1]
                correlations[row, col] = float(np.mean(sample_correlations))

        return correlations

    def _ensure_bmus(self) -> np.ndarray:
        """Return BMU coordinates, calculating and caching them after fit."""
        if self.model is None or self.training_feature_matrix is None:
            raise RuntimeError("fit() must be called before calculating BMUs")
        if self.bmus is None:
            self.bmus = self.model.identify_bmus(
                torch.tensor(self.training_feature_matrix, dtype=torch.float32)
            ).detach().cpu().numpy()
        return self.bmus


class SOM(_SOMCore):
    """Self-organising map analysis and its visualisation diagnostics."""

    def plot_nodes(self, **kwargs: Any):
        if self.model is None or self.hit_map is None or self.feature_data is None:
            raise RuntimeError("fit() must be called before plotting nodes")
        if self.pca_model is not None:
            raise RuntimeError(
                "plot_nodes() is unavailable when pca=True; use "
                "plot_average_assigned_nodes() instead"
            )
        return plot_weekly_snapshot_nodes(
            self.model, self.hit_map, self.feature_data.valid_mask,
            self.feature_data.data.latitude,
            self.feature_data.data.longitude,
            self.feature_data.data.sizes[self.feature_data.data.dims[0]],
            **kwargs,
        )

    def plot_sequence_nodes(self, **kwargs: Any):
        if self.model is None or self.hit_map is None or self.feature_data is None:
            raise RuntimeError("fit() must be called before plotting sequence nodes")
        if self.pca_model is not None:
            raise RuntimeError(
                "plot_sequence_nodes() is unavailable when pca=True; use "
                "plot_average_assigned_nodes() instead"
            )
        return plot_weekly_sequence_nodes(
            self.model, self.hit_map, self.feature_data.valid_mask,
            self.feature_data.data.latitude,
            self.feature_data.data.longitude,
            self.feature_data.data.sizes[self.feature_data.data.dims[0]],
            **kwargs,
        )

    def plot_average_assigned_nodes(
        self, *, figsize: tuple[float, float] | None = None,
        cmap=GHI_CMAP, units_label: str = "GHI", title: str | None = None,
    ):
        """Plot the average prepared sample assigned to each BMU."""
        if self.model is None or self.feature_matrix is None or self.feature_data is None:
            raise RuntimeError("fit() must be called before plotting assigned-node means")
        rows, cols = int(self.model.x), int(self.model.y)
        n_valid = int(self.feature_data.valid_mask.sum())
        values = np.asarray(self.feature_matrix, dtype=float)
        sequence_len, remainder = divmod(values.shape[1], n_valid)
        if remainder:
            raise ValueError("Feature matrix cannot be reconstructed as spatial maps")
        means = np.full((rows, cols, n_valid), np.nan)
        bmus = self._ensure_bmus()
        for row in range(rows):
            for col in range(cols):
                assigned = values[(bmus[:, 0] == row) & (bmus[:, 1] == col)]
                if len(assigned):
                    if sequence_len > 1:
                        assigned = assigned.reshape(len(assigned), sequence_len, n_valid).mean(axis=1)
                    means[row, col] = assigned.mean(axis=0)
        maps = [[self._unflatten_to_grid(means[r, c]) for c in range(cols)] for r in range(rows)]
        fig, axes = plt.subplots(rows, cols, figsize=figsize or (3.2 * cols, 2.8 * rows),
                                 subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
        mesh = None
        for row in range(rows):
            for col in range(cols):
                mesh = GHIPlotter.plot_ghi(maps[row][col], ax=axes[row, col],
                                           extent=GHIPlotter.EXTENT, cmap=cmap,
                                           vmin=GHI_VMIN, vmax=GHI_VMAX, add_colorbar=False)
                for artist in list(axes[row, col].texts):
                    artist.remove()
                axes[row, col].set_title(
                    f"Node ({row},{col}): {int(self.hit_map[row, col])} samples", fontsize=9
                )
        fig.suptitle(title or f"Mean assigned GHI ({rows}x{cols})")
        fig.subplots_adjust(right=0.9, wspace=0.15, hspace=0.4)
        fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03, label=units_label)
        return fig, axes

    def _sample_bmu_correlations(self) -> np.ndarray:
        """Return one Pearson correlation for every valid assigned sample."""
        if self.model is None or self.training_feature_matrix is None:
            raise RuntimeError("fit() must be called before calculating correlations")
        data = self._training_matrix()
        weights = self.model.weights.detach().cpu().numpy()
        result = []
        for sample, bmu in zip(data, self._ensure_bmus()):
            node = weights[tuple(int(value) for value in bmu)]
            if np.std(sample) > 0 and np.std(node) > 0:
                result.append(float(np.corrcoef(sample, node)[0, 1]))
        return np.asarray(result, dtype=float)

    def plot_bmu_correlation_cdf(self, *, ax: Any = None, figsize=(5.5, 4.0), thresholds=None):
        """Plot the empirical CDF of assigned-sample Pearson correlations."""
        correlations = self._sample_bmu_correlations()
        thresholds = np.asarray(thresholds if thresholds is not None else np.linspace(-1, 1, 101))
        cdf = np.array([
            (correlations <= threshold).mean() if len(correlations) else np.nan
            for threshold in thresholds
        ])
        figure, axis = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        axis.plot(thresholds, cdf, color="tab:blue")
        axis.set(xlim=(-1, 1), ylim=(0, 1.02), xlabel="Pearson correlation threshold",
                 ylabel="Fraction at or below threshold", title="CDF of BMU-to-sample Pearson correlation")
        axis.grid(alpha=0.3)
        if ax is None:
            figure.tight_layout()
        return figure, thresholds, cdf

    def plot_bmu_correlations(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] = (4.5, 3.6),
        cmap: str = "RdYlGn",
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot the average node-to-assigned-sample correlation heat map.

        Each cell is labelled with the average Pearson correlation for that
        SOM node. Returns the figure and the ``(rows, cols)`` correlation
        array used to create it.
        """
        correlations = self.average_bmu_correlations()
        figure, axis = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        image = axis.imshow(correlations, cmap=cmap, vmin=vmin, vmax=vmax)
        rows, cols = correlations.shape
        for row in range(rows):
            for col in range(cols):
                value = correlations[row, col]
                label = "—" if np.isnan(value) else f"{value:.2f}"
                axis.text(col, row, label, ha="center", va="center", fontsize=9)
        axis.set_xticks(range(cols))
        axis.set_yticks(range(rows))
        axis.set_xlabel("node col")
        axis.set_ylabel("node row")
        axis.set_title("Average BMU-to-sample\nPearson correlation")
        figure.colorbar(image, ax=axis, label="Pearson r")
        if ax is None:
            figure.tight_layout()
        return figure, correlations

    def plot_bmu_timeline(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] | None = None,
        year_by_year: bool = True,
    ) -> tuple[plt.Figure, dict[tuple[int, int], list[Any]]]:
        """Plot the BMU assignment of every prepared sample across time.

        Each row represents one SOM node. A coloured cell marks a sample
        assigned to that node; white cells indicate that the sample belongs
        to another node. Returns the figure and a mapping from ``(row, col)``
        node identifiers to the timestamps or sample coordinates assigned to
        each node.
        """
        if self.model is None or self.feature_matrix is None or self.feature_data is None:
            raise RuntimeError("fit() must be called before plotting BMU assignments")

        sample_dim = self.feature_data.data.dims[0]
        sample_coord = self.feature_data.data.coords.get(sample_dim)
        if sample_coord is None:
            raise ValueError(f"Prepared data has no coordinate for sample dimension {sample_dim!r}")
        raw_timestamps = sample_coord.values
        timestamps = [
            value.item() if isinstance(value, np.generic) else value
            for value in raw_timestamps
        ]

        bmus = self._ensure_bmus()
        rows, cols = int(self.model.x), int(self.model.y)
        nodes = [(row, col) for row in range(rows) for col in range(cols)]
        node_indices = {node: index for index, node in enumerate(nodes)}
        assignments = {
            node: [timestamps[index] for index, bmu in enumerate(bmus)
                   if tuple(int(value) for value in bmu) == node]
            for node in nodes
        }

        matrix = np.zeros((len(nodes), len(timestamps)), dtype=int)
        for sample_index, bmu in enumerate(bmus):
            node = tuple(int(value) for value in bmu)
            matrix[node_indices[node], sample_index] = node_indices[node] + 1

        node_colours = plt.colormaps["turbo"].resampled(max(len(nodes), 1))(np.arange(len(nodes)))
        cmap = ListedColormap(np.vstack(([1.0, 1.0, 1.0, 1.0], node_colours)))
        norm = BoundaryNorm(np.arange(-0.5, len(nodes) + 1.5), len(nodes) + 1)

        date_values: pd.DatetimeIndex | None = None
        try:
            if sample_dim == "dayofyear":
                date_values = pd.Timestamp("2001-01-01") + pd.to_timedelta(
                    np.asarray(raw_timestamps, dtype=int) - 1, unit="D"
                )
            else:
                date_values = pd.DatetimeIndex(pd.to_datetime(raw_timestamps))
        except (TypeError, ValueError, OverflowError):
            date_values = None

        if year_by_year and date_values is not None and len(date_values):
            years = date_values.year.to_numpy()
            if self.config.season == "DJF":
                years = np.where(date_values.month.to_numpy() == 12, years + 1, years)
            panel_keys = sorted(set(int(year) for year in years))
            panel_indices = [np.flatnonzero(years == year) for year in panel_keys]
        else:
            panel_keys = [None]
            panel_indices = [np.arange(len(timestamps))]

        if ax is not None:
            if len(panel_indices) != 1:
                raise ValueError("An explicit ax can only be used with year_by_year=False")
            figure = ax.figure
            axes = [ax]
        else:
            height = max(4, 0.35 * len(nodes)) * len(panel_indices)
            figure, panel_axes = plt.subplots(
                len(panel_indices), 1,
                figsize=figsize or (14, height),
                sharey=True,
                squeeze=False,
                constrained_layout=True,
            )
            axes = list(panel_axes[:, 0])

        image = None
        for panel_axis, panel_key, indices in zip(axes, panel_keys, panel_indices):
            panel_matrix = matrix[:, indices]
            image = panel_axis.imshow(
                panel_matrix,
                aspect="auto",
                interpolation="none",
                cmap=cmap,
                norm=norm,
            )
            panel_axis.set_yticks(np.arange(len(nodes)))
            panel_axis.set_yticklabels([f"({row}, {col})" for row, col in nodes])
            panel_axis.set_ylabel("SOM node")
            panel_axis.set_title(
                f"Season year {panel_key}" if panel_key is not None else "BMU assignments"
            )

            if date_values is not None and len(indices):
                panel_dates = date_values[indices]
                month_starts = pd.date_range(
                    panel_dates.min().to_period("M").to_timestamp(),
                    panel_dates.max().to_period("M").to_timestamp(),
                    freq="MS",
                )
                positions = [
                    int(np.argmin(np.abs(panel_dates.asi8 - month_start.value)))
                    for month_start in month_starts
                ]
                unique_positions = sorted(set(positions))
                panel_axis.set_xticks(unique_positions)
                panel_axis.set_xticklabels(
                    [panel_dates[position].strftime("%b\n%Y") for position in unique_positions],
                    rotation=0 if len(unique_positions) <= 4 else 45,
                    ha="right" if len(unique_positions) > 4 else "center",
                )
                panel_axis.set_xlabel("Training date")
            else:
                panel_axis.set_xlabel("Training sample")

        if image is None:
            raise RuntimeError("No samples available for BMU timeline")
        colorbar = figure.colorbar(
            image,
            ax=axes,
            ticks=np.arange(1, len(nodes) + 1),
            orientation="horizontal",
            fraction=0.03,
            pad=0.08,
        )
        colorbar.ax.set_xticklabels([f"({row}, {col})" for row, col in nodes])
        colorbar.set_label("BMU node")
        if len(panel_indices) > 1:
            figure.suptitle("BMU assignments by year", y=1.0)
        if ax is not None:
            figure.tight_layout()
        return figure, assignments


class SOMGridSearch(SOM):
    """SOM workflow for evaluating multiple grid shapes."""

    def run_grid_search(
        self,
        shapes: list[tuple[int, int]] | None = None,
        n_init: int = 3,
        epochs: int = EPOCHS,
        output_path: str | Path | None = None,
    ):
        """Run the SOM grid-size search on the prepared feature matrix."""
        X = self.feature_matrix if self.feature_matrix is not None else self.prepare()
        return som_grid_search(
            X,
            shapes=shapes,
            n_init=n_init,
            epochs=epochs,
            config=self.config,
            output_path=output_path,
        )


def load_climatology(ghi_dir: str | Path, average_years: bool = False, regrid: bool = True) -> xr.DataArray:
    return DataLoader.from_file(
        ghi_dir,
        regrid_equal_area=regrid,
    ).climatology(average_years=average_years)

def train_som(X: np.ndarray, rows: int, cols: int, verbose: bool = True, plot_errors=False, output_dir=None,  **som_kwargs) -> tuple[TorchSOM, torch.Tensor, np.ndarray, dict]:
    """Trains a torchsom SOM with a rows x cols grid on feature matrix X
    (n_samples x n_features), using this module's shared hyperparameters
    (overridable via som_kwargs), on MPS if available. Grid shape is always
    an explicit argument here (not a module default) - callers such as
    main() and som_grid_search decide it themselves.

    Returns (som, data, hit_map, info_dict) - hit_map is a (rows, cols) ndarray of how
    many of X's rows landed on each node, info_dict summarises training specs"""
    kwargs: dict[str, Any] = dict(DEFAULT_TRAINING_ARGS)

    kwargs.update(som_kwargs)
    kwargs["x"] = rows
    kwargs["y"] = cols

    data = torch.tensor(X, dtype=torch.float32)
    som = TorchSOM(num_features=X.shape[1], **kwargs)
    som.initialize_weights(data, mode=kwargs["initialization_mode"])
    q_errors, t_errors = som.fit(data, verbose=False)
    if plot_errors:
        config = VisualizationConfig(
                figsize=(12, 8),                       # figure size in inches
                fontsize={"title": 16, "axis": 13, "legend": 11},
                fontweight={"title": "bold", "axis": "normal", "legend": "normal"},
                cmap=GHI_CMAP,                           # cmocean solar
                dpi=300,                               # resolution for saved figures
                grid_alpha=0.3,                        # grid transparency
                colorbar_pad=0.01,                     # colorbar padding
                save_format="png",                     # png, pdf, eps, or svg
                hex_radius=0.5,                        # hexagon radius (hexagonal topology)
                hex_border_color="black",
                hex_border_width=0.3,
            )
        viz = SOMVisualizer(som=som, config=config)
        if output_dir is not None:
            viz.plot_training_errors(
                quantization_errors=q_errors,
                topographic_errors=t_errors,
                save_path=output_dir
                )
        else:   
            viz.plot_training_errors(
                quantization_errors=q_errors,
                topographic_errors=t_errors,
                )

    hit_map = som.build_map("hit", data=data).cpu().numpy()
    dead = int((hit_map == 0).sum())
    if dead:
        warnings.warn(
            f"train_som: {dead}/{hit_map.size} node(s) got zero hits "
            f"out of {X.shape[0]} samples - map may be under-trained or "
            f"oversized for this few samples; inspect the hit map",
            UserWarning,
            stacklevel=2,
        )
    if verbose:
        print(
            f"train_som: final quantization error {q_errors[-1]:.3f}, "
            f"topographic error {t_errors[-1]:.2f}%"
        )
    return som, data, hit_map, kwargs

def som_grid_search(
    X: np.ndarray,
    shapes: list[tuple[int, int]] | None = None,
    n_init: int = 1,
    epochs: int = EPOCHS,
    *,
    config: DataConfig | None = None,
    output_path: str | Path | None = None,
):
    """Sweeps candidate SOM grid shapes on feature matrix X, Each shape is trained n_init times (random_seed 0..n_init-1)
    and errors averaged. Also reports, from the random_seed=0 run alone, dead
    node count and mean hits/node - with few samples (this project's weekly
    climatologies top out around 52), a grid growing faster than the sample
    count starves nodes of data well before the error curves show it.

    Returns (fig, results) where results is a list of dicts, one per shape,
    each with rows, cols, n_nodes, quantization_error, topographic_error
    (means over n_init runs), dead_nodes and mean_hits_per_node (from the
    single random_seed=0 run).

    Example Usage:
    fig, results = som_grid_search(X1)
    fig, results = som_grid_search(X1, shapes=[(2, 2), (3, 3), (4, 4), (5, 5)])"""
    if shapes is None:
        shapes = [(4, 4), (5, 5), (6, 6), (8,8)]

    results = []
    for rows, cols in tqdm(shapes):
        q_errors, t_errors = [], []
        dead_nodes = mean_hits = None
        grid_warnings: list[str] = []
        for seed in range(n_init):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                som, data, hit_map, _training_info = train_som(
                    X, rows, cols, epochs=200, random_seed=seed, verbose=False, plot_errors=False
                )
            grid_warnings.extend(str(item.message) for item in caught)
            q_errors.append(som.quantization_error(data))
            t_errors.append(som.topographic_error(data))
            if seed == 0:
                dead_nodes = int((hit_map == 0).sum())
                mean_hits = X.shape[0] / hit_map.size
        results.append(
            dict(
                grid_size=[rows, cols],
                rows=rows, cols=cols, n_nodes=rows * cols,
                quantization_error=float(np.mean(q_errors)),
                topographic_error=float(np.mean(t_errors)),
                dead_nodes=dead_nodes, mean_hits_per_node=mean_hits,
                warnings=grid_warnings,
            )
        )
        print(
            f"som_grid_search: {rows}x{cols} ({rows * cols} nodes) - "
            f"QE {results[-1]['quantisation_error']:.3f}, "
            f"TE {results[-1]['topographic_error']:.2f}%, "
            f"dead nodes {dead_nodes}/{rows * cols}, "
            f"mean hits/node {mean_hits:.1f}"
        )

    n_nodes = [r["n_nodes"] for r in results]
    fig, qe_ax = plt.subplots(figsize=(6, 4))
    qe_ax.plot(n_nodes, [r["quantization_error"] for r in results], "o-", color="tab:blue")
    qe_ax.set_xlabel("Number of nodes")
    qe_ax.set_ylabel(f"Quantization error (mean of {n_init} runs)", color="tab:blue")
    qe_ax.tick_params(axis="y", labelcolor="tab:blue")
    qe_ax.set_title(f"SOM grid-size elbow ({X.shape[0]} samples, {X.shape[1]} features)")

    te_ax = qe_ax.twinx()
    te_ax.plot(n_nodes, [r["topographic_error"] for r in results], "s--", color="tab:red")
    te_ax.set_ylabel(f"Topographic error % (mean of {n_init} runs)", color="tab:red")
    te_ax.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "analysis": asdict(config) if config is not None else None,
                "n_init": n_init,
                "epochs": epochs,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            },
            "results": results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    return fig, results

def json_date(timestamp: Any) -> str:
    """Format a cached timestamp for JSON without changing BMU plot data."""
    if isinstance(timestamp, (int, np.integer)):
        return pd.Timestamp(timestamp, unit="ns").strftime("%d-%m-%Y")
    return pd.Timestamp(timestamp).strftime("%d-%m-%Y")

def run_seasonal_som(
    ghi_dir: str | Path,
    season: Literal["DJF", "MAM", "JJA", "SON"],
    rows: int,
    cols: int,
    output_dir: str | Path | None = None,
    *,
    aggregate_years: bool = False,
    scale: bool = False,
    representation: Literal["daily", "weekly", "weekly_mean", "weekly_sequence"] = "daily",
    overlap_sequences: bool = True,
    seq_len: int = 7,
    pca: bool = False,
    pca_threshold: float | None = None,
    pca_max_ncomp: int = 100,
    som_training_args: dict[str, Any] | None = None,
    comparison_mode: Literal["paired", "real_samples"] = "paired",
) -> SOM:
    """Run a seasonal SOM and save its standard diagnostic visualisations.

    ``ghi_dir`` is required. ``output_dir`` must be an absolute path; when
    omitted or relative, a warning is issued and the current working
    directory is used. The saved outputs are the SOM node maps, BMU
    timeline, average BMU-correlation heat map, node-versus-closest-assigned
    real-sample comparison, and a JSON file containing the configuration,
    training metrics, hit map, correlations, and warnings.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if comparison_mode == "real_samples" and representation != "weekly_sequence":
        raise ValueError("comparison_mode='real_samples' requires weekly_sequence representation")

    if output_dir is None or not Path(output_dir).is_absolute():
        warnings.warn("output_dir should be a full path; using the current working directory", UserWarning, stacklevel=2)
        output_path = Path.cwd()
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    season_name = season.lower()
    config = DataConfig(
        regrid_equal_area=True,
        aggregate_years=aggregate_years,
        season=season,
        scale=scale,
        representation=representation,
        overlap_sequences=overlap_sequences,
        seq_len=seq_len,
        pca=pca,
        pca_threshold=pca_threshold,
        pca_max_ncomp=pca_max_ncomp,
    )

    with tqdm(total=3, desc=f"{season_name.title()} SOM", unit="stage") as progress:
        progress.set_description("Loading data")
        data = DataLoader.from_file(ghi_dir, regrid_equal_area=config.regrid_equal_area)
        progress.update(1)

        progress.set_description("Preparing and training SOM")
        som = SOM(data, config).fit(
            rows,
            cols,
            output_dir=output_path,
            **(som_training_args or {}),
        )
        progress.update(1)

        progress.set_description("Writing outputs")
        units_label = "GHI (scaled)" if scale else "GHI"
        sample_label = "day" if representation == "daily" else "week"
        if config.pca:
            nodes_figure, _ = som.plot_average_assigned_nodes(
                units_label=units_label,
                title=f"{season} SOM mean assigned samples ({rows}x{cols}, PCA)",
            )
            nodes_filename = f"{season_name}_{rows}x{cols}_average_assigned_nodes.png"
        elif representation == "weekly_sequence":
            nodes_figure = som.plot_sequence_nodes(units_label=units_label)
            nodes_filename = f"{season_name}_{rows}x{cols}_sequence_nodes.png"
        else:
            nodes_figure, _ = som.plot_nodes(
                units_label=units_label,
                sample_label=sample_label,
                title=f"{season} SOM nodes ({rows}x{cols})",
            )
            nodes_filename = f"{season_name}_{rows}x{cols}_nodes.png"
        nodes_figure.savefig(output_path / nodes_filename, dpi=150, bbox_inches="tight")
        plt.close(nodes_figure)

        if not config.pca:
            average_figure, _ = som.plot_average_assigned_nodes(
                units_label=units_label,
                title=f"{season} SOM mean assigned samples ({rows}x{cols})",
            )
            average_figure.savefig(
                output_path / f"{season_name}_{rows}x{cols}_average_assigned_nodes.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(average_figure)

        timeline_figure, assignments = som.plot_bmu_timeline(year_by_year=True)
        timeline_figure.savefig(
            output_path / f"{season_name}_{rows}x{cols}_bmu_timeline.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(timeline_figure)

        correlation_figure, correlations = som.plot_bmu_correlations(figsize=(7, 5))
        correlation_figure.savefig(
            output_path / f"{season_name}_{rows}x{cols}_bmu_correlations.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(correlation_figure)

        cdf_figure, _, _ = som.plot_bmu_correlation_cdf()
        cdf_figure.savefig(output_path / f"{season_name}_{rows}x{cols}_bmu_correlation_cdf.png",
                           dpi=150, bbox_inches="tight")
        plt.close(cdf_figure)

        bmus = som._ensure_bmus()
        chosen_samples: dict[tuple[int, int], int] = {}
        if not config.pca:
            X = som._training_matrix()
            training_weights = som.model.weights.detach().cpu().numpy()
            weights = som.node_weights_in_original_space()
            sample_dim = som.feature_data.data.dims[0]
            sample_values = som.feature_data.data.coords[sample_dim].values
            sample_labels = [str(value) for value in sample_values]
            for row in range(rows):
                for col in range(cols):
                    indices = np.flatnonzero(
                        (bmus[:, 0] == row) & (bmus[:, 1] == col)
                    )
                    if len(indices):
                        distances = np.linalg.norm(
                            X[indices] - training_weights[row, col], axis=1
                        )
                        chosen_samples[(row, col)] = int(indices[np.argmin(distances)])

        if not config.pca and comparison_mode == "real_samples":
            sequence_len = config.seq_len
            n_valid = int(som.feature_data.valid_mask.sum())
            n_nodes = rows * cols
            comparison_figure, axes = plt.subplots(
                n_nodes,
                sequence_len,
                figsize=(2.8 * sequence_len, 2.2 * n_nodes),
                subplot_kw={"projection": ccrs.PlateCarree()},
                squeeze=False,
            )
            for row in range(rows):
                for col in range(cols):
                    node = row * cols + col
                    sample_index = chosen_samples.get((row, col))
                    if sample_index is None:
                        continue
                    sequence = X[sample_index].reshape(sequence_len, n_valid)
                    for step in range(sequence_len):
                        axis = axes[node, step]
                        GHIPlotter.plot_ghi(
                            som._unflatten_to_grid(sequence[step]),
                            ax=axis,
                            extent=GHIPlotter.EXTENT,
                            cmap=GHI_CMAP,
                            vmin=GHI_VMIN,
                            vmax=GHI_VMAX,
                            add_colorbar=False,
                        )
                        axis.set_title(
                            f"({row},{col}) day {step + 1}" if step == 0 else f"day {step + 1}",
                            fontsize=8,
                        )
                        for text_artist in list(axis.texts):
                            text_artist.remove()
                        axis.set_xticks([])
                        axis.set_yticks([])
                    axes[node, 0].set_ylabel(
                        f"{sample_labels[sample_index]}", fontsize=8
                    )
            comparison_figure.suptitle(
                f"{season} SOM: closest real {sequence_len}-day sample per node",
                fontsize=13,
            )
            comparison_filename = f"{season_name}_{rows}x{cols}_real_samples.png"
        elif not config.pca:
            comparison_figure, axes = plt.subplots(
                rows,
                cols * 2,
                figsize=(3.0 * cols * 2, 2.8 * rows),
                subplot_kw={"projection": ccrs.PlateCarree()},
                squeeze=False,
            )
            for row in range(rows):
                for col in range(cols):
                    node_axis, sample_axis = axes[row, 2 * col:2 * col + 2]
                    GHIPlotter.plot_ghi(
                        som._unflatten_to_grid(weights[row, col]),
                        ax=node_axis,
                        extent=GHIPlotter.EXTENT,
                        cmap=GHI_CMAP,
                        vmin=GHI_VMIN,
                        vmax=GHI_VMAX,
                        add_colorbar=False,
                    )
                    node_axis.set_title(f"Node ({row},{col})", fontsize=8)
                    if (row, col) in chosen_samples:
                        sample_index = chosen_samples[(row, col)]
                        GHIPlotter.plot_ghi(
                            som._unflatten_to_grid(X[sample_index]),
                            ax=sample_axis,
                            extent=GHIPlotter.EXTENT,
                            cmap=GHI_CMAP,
                            vmin=GHI_VMIN,
                            vmax=GHI_VMAX,
                            add_colorbar=False,
                        )
                        sample_axis.set_title(
                            f"Real sample\n{sample_labels[sample_index]}", fontsize=8
                        )
                    else:
                        sample_axis.set_title("No assigned sample", fontsize=8)
                        sample_axis.axis("off")
                    for axis in (node_axis, sample_axis):
                        for text_artist in list(axis.texts):
                            text_artist.remove()
                        axis.set_xticks([])
                        axis.set_yticks([])
            comparison_figure.suptitle(
                f"{season} SOM nodes compared with closest assigned real samples",
                fontsize=13,
            )
            comparison_filename = f"{season_name}_{rows}x{cols}_node_vs_real_samples.png"
        if not config.pca:
            comparison_figure.tight_layout()
            comparison_figure.savefig(
                output_path / comparison_filename,
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(comparison_figure)

        run_info = {
            "config": asdict(config),
            "grid_size": [rows, cols],
            "training_metrics": som.training_metrics,
            "warnings": som.warnings,
            "hit_map": som.hit_map.tolist(),
            "average_bmu_correlations": correlations.tolist(),
            "bmu_assignments": {
                f"({row},{col})": [json_date(timestamp) for timestamp in timestamps]
                for (row, col), timestamps in assignments.items()
            },
            "comparison_sample_indices": {
                f"({row},{col})": {
                    "index": index,
                    "timestamp": json_date(sample_values[index]),
                }
                for (row, col), index in chosen_samples.items()
            },
        }
        with open(output_path / "specs.json", "w") as handle:
            json.dump(run_info, handle, indent=2, default=str)
        progress.update(1)
        progress.set_description("Complete")

    return som

def main(
    ghi_dir: str | Path,
    config: DataConfig | None = None,
    rows: int = 3,
    cols: int = 3,
    som_training_args: dict | None= None,
    output_dir: str | Path | None = None,
) -> SOM:
    """Run the configured seasonal SOM and save all standard diagnostics.

    This is the public entry point for the full workflow: learned node maps,
    mean assigned-sample maps, BMU timeline, average-correlation heatmap,
    correlation threshold CDF, and node-versus-real-sample comparison.
    """
    config = config or DataConfig(season="DJF")
    season = config.season or "DJF"
    return run_seasonal_som(
        ghi_dir,
        season,
        rows,
        cols,
        output_dir,
        aggregate_years=config.aggregate_years,
        scale=config.scale,
        representation=config.representation,
        overlap_sequences=config.overlap_sequences,
        seq_len=config.seq_len,
        som_training_args=som_training_args,
    )

if __name__ == "__main__":
    # main(
    #     config=DataConfig(
    #         regrid_equal_area=True,
    #         aggregate_years=False,
    #         season='DJF',
    #         scale=False,
    #         representation='daily'
    #     ),
    #     rows=3,
    #     cols=3,
    #     output_dir="summer_results"
    # )

    # config = DataConfig(season="DJF", representation='weekly_sequence', overlap_sequences=True)
    # main_grid_search("/full/path/to/ghi", config, "/full/path/to/results", shapes = [(2,2),(2,3),(3,3),(3,4),(4,4),(4,5),(5,5)])

    names = {"DJF":"summer", "MAM":"autumn", "JJA":"winter", "SON":"spring"}
    sizes = {"DJF":(3,3), "MAM":(3,4), "JJA":(3,3), "SON":(3,3)}
    # params = [(5,0.9), (5,0.5), (2,0.9), (2,0.5), (1.5,0.9), (1.5,0.5), (1.2, 0.9), (1.2, 0.5)]
    # for season, size in sizes.items():
    #     rows, cols = size
    #     dir = f"{names[season]}_results/{rows}x{cols}_pca"
    #     run_seasonal_som("/full/path/to/ghi", season, rows, cols, dir, 
    #                     som_training_args={"epochs":300, "plot_errors":True, "initialization_mode":"pca"})
        # print(f"finished {season}")


    # Run explicitly, for example:
    # run_seasonal_som(
    #     "/full/path/to/ghi",
    #     season="DJF",
    #     rows=3,
    #     cols=3,
    #     output_dir="/full/path/to/results",
    #     representation="daily",
    #     som_training_args={"epochs": 300, "plot_errors": True},
    # )
