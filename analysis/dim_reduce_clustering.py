"""PCA + conventional clustering, as an alternative to som.py's SOM: reduce
each weekly/daily GHI sample down to just enough principal components to
explain a target fraction of total variance (default 90%), cluster the
reduced samples with KMeans or hierarchical clustering, and display each
cluster's mean GHI spatial pattern back on the map.

Uses ``DataLoader`` and ``AnalysisBase`` for data loading and preparation,
and ``clustering.elbow_silhouette_analysis`` for selecting a useful cluster
count diagnostically. PCA reduction makes the feature matrix smaller before
clustering while spatial cluster means are reconstructed from the original
prepared features.
"""

from typing import Any, Literal
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

from .analysis import AnalysisBase, DataConfig
from .clustering import elbow_silhouette_analysis
from .dataloader import DataLoader
from .plotting import GHIPlotter, GHI_CMAP, GHI_VMIN, GHI_VMAX

_ENGINES = {
    "kmeans": KMeans,
    "hierarchical": AgglomerativeClustering,
}
DEFAULT_CONFIGS = {
    "kmeans":{"n_clusters": 9,
              "init": "k-means++",
              "random_state":0},

    "hierarchical":{"n_clusters":9,
                    "metric":"euclidean",
                    "linkage":"ward"
                    }
}

MAX_COMPONENTS = 100
GHI_DIR = Path("aus_ghi")
OUTPUT_DIR = Path("results")


class PCAClustering(AnalysisBase):
    def __init__(self, data: Any, config: DataConfig | None = None) -> None:
        super().__init__(data, config)
        self.pca: PCA | None = None
        self.reduced_features: np.ndarray | None = None
        self.n_components: int | None = None
        self.cluster_model: KMeans | AgglomerativeClustering | None = None
        self.labels: np.ndarray | None = None


    def reduce_dimensions(
        self,
        variance_threshold: float = 0.90,
        *,
        max_components: int = MAX_COMPONENTS,
    ) -> tuple[np.ndarray, PCA, int]:
        """Runs PCA on X (n_samples x n_features) and keeps just enough leading
        components to explain at least variance_threshold of total variance.

        Returns (X_reduced, pca, n_components):
        - X_reduced: ndarray (n_samples, n_components)
        - pca: the fitted sklearn PCA (up to ``max_components`` components,
            for inspecting explained_variance_ratio_ etc.)
        - n_components: how many of those were actually kept

        Example Usage:
        X_reduced, pca, n_components = reduce_dimensions(X1, variance_threshold=0.9)"""
        if not 0 < variance_threshold <= 1:
            raise ValueError("variance_threshold must be in the interval (0, 1]")
        if max_components <= 0:
            raise ValueError("max_components must be positive")

        X = self.feature_matrix if self.feature_matrix is not None else self.prepare()
        fit_components = min(max_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=fit_components, random_state=0).fit(X)
        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components = int(
            min(np.searchsorted(cumulative, variance_threshold) + 1, fit_components)
        )
        self.reduced_features = self.pca.transform(X)[:, :self.n_components]
        return self.reduced_features, self.pca, self.n_components

    def scree_plot(
        self,
        *,
        ax: Any = None,
        figsize: tuple[float, float] = (8.0, 5.0),
        variance_threshold: float | None = None,
    ) -> tuple[plt.Figure, Any]:
        """Plot individual and cumulative explained variance for fitted PCA.

        ``reduce_dimensions`` must be called first. If ``variance_threshold``
        is supplied, the selected component count is marked on the plot.
        Returns the figure and axes used for the plot.
        """
        if self.pca is None:
            raise RuntimeError("reduce_dimensions() must be called before scree_plot()")

        if ax is None:
            figure, axis = plt.subplots(figsize=figsize)
        else:
            figure, axis = ax.figure, ax
        components = np.arange(1, len(self.pca.explained_variance_ratio_) + 1)
        explained = self.pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        axis.bar(components, explained, alpha=0.65, label="Individual variance")
        axis.plot(components, cumulative, "o-", color="black", label="Cumulative variance")
        if variance_threshold is not None:
            if not 0 < variance_threshold <= 1:
                raise ValueError("variance_threshold must be in the interval (0, 1]")
            selected = int(min(np.searchsorted(cumulative, variance_threshold) + 1, len(components)))
            axis.axvline(selected, color="tab:red", linestyle="--", label=f"Selected: {selected}")
            axis.axhline(variance_threshold, color="tab:red", linestyle=":")
        axis.set_xlabel("Principal component")
        axis.set_ylabel("Explained variance ratio")
        axis.set_title("PCA scree plot")
        axis.set_xticks(components)
        axis.legend()
        if ax is None:
            figure.tight_layout()
        return figure, axis


    def cluster(
        self,
        engine: Literal["kmeans", "hierarchical"] = "kmeans",
        engine_config: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Fit the selected clustering engine to the reduced PCA features."""
        if engine not in _ENGINES:
            raise ValueError(f"Invalid cluster engine {engine!r}; choose from {list(_ENGINES)}")
        if self.reduced_features is None:
            raise RuntimeError("reduce_dimensions() must be called before cluster()")

        config = dict(DEFAULT_CONFIGS[engine])
        if engine_config is not None:
            config.update(engine_config)
        self.cluster_model = _ENGINES[engine](**config).fit(self.reduced_features)
        self.labels = np.asarray(self.cluster_model.labels_, dtype=int)
        return self.labels

    def elbow_analysis(
        self,
        cluster_range: range | list[int] = range(2, 11),
        *,
        n_init: int = 3,
    ) -> tuple[plt.Figure, plt.Figure]:
        """Run KMeans elbow and silhouette diagnostics on reduced features."""
        if self.reduced_features is None:
            raise RuntimeError("reduce_dimensions() must be called before elbow_analysis()")
        return elbow_silhouette_analysis(
            self.reduced_features,
            cluster_range=cluster_range,
            n_init=n_init,
        )


    def plot_cluster_means(
        self,
        *,
        cmap=GHI_CMAP,
        units_label: str = "GHI",
        sample_label: str = "week",
        ncols: int = 3,
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Displays each cluster's mean GHI spatial pattern - the average, in
        the original un-reduced feature space (not PCA space), of every sample
        assigned to that cluster - in a grid of maps. Analogue of
        som.plot_weekly_snapshot_nodes, but for conventional cluster labels
        instead of SOM node weight vectors."""
        if self.labels is None or self.feature_data is None or self.feature_matrix is None:
            raise RuntimeError("cluster() must be called before plot_cluster_means()")
        if ncols <= 0:
            raise ValueError("ncols must be positive")
        if self.feature_matrix.shape[1] != int(self.feature_data.valid_mask.sum()):
            raise ValueError("Cluster mean maps require daily or weekly snapshot data, not sequences")

        X = self.feature_matrix
        labels = self.labels
        valid = self.feature_data.valid_mask
        n_samples = X.shape[0]
        unique_labels = sorted(set(labels.tolist()))
        n_clusters = len(unique_labels)
        nrows = -(-n_clusters // ncols)  # ceil division

        cluster_maps = []
        counts = []
        for k in unique_labels:
            mask = labels == k
            counts.append(int(mask.sum()))
            cluster_maps.append(self._unflatten_to_grid(X[mask].mean(axis=0)))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
        )
        axes_flat = axes.flat
        mesh = None
        for k, cluster_map, count, ax in zip(unique_labels, cluster_maps, counts, axes_flat):
            mesh = GHIPlotter.plot_ghi(
                cluster_map, ax=ax, extent=GHIPlotter.EXTENT, cmap=cmap,
                vmin=GHI_VMIN, vmax=GHI_VMAX, add_colorbar=False,
            )
            for txt in list(ax.texts):
                txt.remove()  # drop plot_ghi's per-axes min/max annotation - too busy at this size
            ax.set_title(
                f"Cluster {k}: {count}/{n_samples} {sample_label}s ({100 * count / n_samples:.0f}%)",
                fontsize=9,
            )
        for ax in list(axes_flat)[n_clusters:]:
            ax.axis("off")

        fig.suptitle(title or f"Cluster means ({n_clusters} clusters)", fontsize=13)
        fig.subplots_adjust(right=0.9, wspace=0.15, hspace=0.4)
        cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
        fig.colorbar(mesh, cax=cbar_ax, label=units_label)
        return fig, axes





def main(
    scale_by_season: bool = False,
    average_years: bool = False,
    daily: bool = False,
    variance_threshold: float = 0.90,
    n_clusters: int = 6,
    engine_name: Literal["kmeans", "hierarchical"] = "kmeans",
    cluster_range: range | list[int] = range(2, 11),
    output_dir: str | None = None,
    season: Literal["DJF", "MAM", "JJA", "SON"] | None = None,
) -> PCAClustering:
    """Runs the PCA + clustering pipeline end to end: load data (same
    toggles as som.main/run_daily_som - scale_by_season, average_years,
    daily), PCA down to variance_threshold, an elbow/silhouette sweep over
    cluster_range (clustering.elbow_silhouette_analysis, diagnostic only -
    doesn't pick n_clusters automatically, same as that function's existing
    usage in clustering.py), then a final fit with n_clusters using
    engine_name ("kmeans" or "hierarchical")."""

    base = output_dir or ("dim_reduce_outputs_daily" if daily else "dim_reduce_outputs")
    output_path = Path(base)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    units_label = "GHI (seasonal z-score)" if scale_by_season else "GHI"
    print(
        f"scale_by_season={scale_by_season}, average_years={average_years}, "
        f"daily={daily}, output_dir={output_path}/"
    )

    config = DataConfig(
        aggregate_years=average_years,
        season=season,
        scale=scale_by_season,
        representation="daily" if daily else "weekly_mean",
    )
    data = DataLoader.from_file(GHI_DIR, regrid_equal_area=config.regrid_equal_area)
    analysis = PCAClustering(data, config)
    sample_label = "day" if daily else "week"
    n_samples = int(analysis.feature_matrix.shape[0])
    valid = analysis.feature_data.valid_mask
    print(f"{n_samples} {sample_label}s, {int(valid.sum())}/{valid.size} locations with data")

    print(f"\nRunning PCA (target >= {variance_threshold:.0%} variance)...")
    _X_reduced, _pca, n_components = analysis.reduce_dimensions(variance_threshold)
    scree_fig, _ = analysis.scree_plot(variance_threshold=variance_threshold)
    scree_fig.savefig(output_path / "scree.png", dpi=150, bbox_inches="tight")

    print("\nRunning elbow/silhouette analysis over cluster_range (KMeans, diagnostic only)...")
    elbow_fig, silhouette_fig = analysis.elbow_analysis(cluster_range)
    elbow_fig.savefig(output_path / "elbow.png", dpi=150, bbox_inches="tight")
    silhouette_fig.savefig(output_path / "silhouette.png", dpi=150, bbox_inches="tight")

    print(f"\nFitting {engine_name} with n_clusters={n_clusters} on {n_components} PCs...")
    analysis.cluster(
        engine=engine_name,
        engine_config={"n_clusters": n_clusters},
    )

    fig, _axes = analysis.plot_cluster_means(
        units_label=units_label, sample_label=sample_label,
        title=f"{engine_name} cluster means (k={n_clusters}, {n_components} PCs, {sample_label}ly)",
    )
    fig.savefig(output_path / "cluster_means.png", dpi=150, bbox_inches="tight")

    print(f"\nFigures saved under {output_path}/")
    plt.show()
    return analysis


if __name__ == "__main__":
    # main(output_dir="PCA_results_02-09", season="JJA", scale_by_season=True)
    from .som import SEASONS
    # for season in SEASONS:
    #     data = DataLoader.from_file("aus_ghi")
    #     config = DataConfig(season=season)
    #     pca = PCAClustering(data, config)
    #     pca.reduce_dimensions(max_components=20)
    #     fig, _ = pca.scree_plot()
    #     fig.savefig(f"results/pca/{season}_scree_plot")

    data = DataLoader.from_file("aus_ghi")
    config = DataConfig(season="DJF")
    pca = PCAClustering(data, config)
    pca.reduce_dimensions(max_components=10)
    pca.cluster('kmeans', {"n_clusters":12})
    fig, _ = pca.plot_cluster_means()
    fig.savefig("results/pca/summer_cluster_means")
    plt.show()
