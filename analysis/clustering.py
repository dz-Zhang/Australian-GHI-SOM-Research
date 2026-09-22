import argparse
import xarray as xr
import numpy as np
from .dataloader import open_ghi
from .plotting import AUSTRALIA_EXTENT, GHI_CMAP, format_labels, plot_ghi
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import cartopy.crs as ccrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="Filename to process")
    parser.add_argument("-station", action="store_true",
                         help="Treat file as a single station time series: print summary stats and plot it")
    args = parser.parse_args()

    if args.station:
        if not args.file:
            parser.error("-station requires a file argument")
        path = Path(args.file)
        stats = summary_stats(path)
        print(stats)
        plt.show()
        return

    input = Path('/Users/jinyaozhang/Downloads/UNSW Research/Code/aus_ghi/ghi_20240101_20241231.nc')
    make_gif(input, weekly=False)


def summary_stats(path: Path) -> dict:
    """Loads a single station time series (as written by open_station_data /
    open_stations_data - just a time-indexed DataArray, no spatial dims),
    prints basic stats and plots the series.
    Returns the stats as a dict."""
    with xr.open_dataarray(path, engine="netcdf4") as da:
        da = da.load()

    stats = {
        "mean": float(da.mean()),
        "std": float(da.std()),
        "min": float(da.min()),
        "max": float(da.max()),
        "count": int(da.count()),
    }

    fig, ax = plt.subplots()
    da.plot.line(ax=ax)
    ax.set_title(f"GHI Time Series: {path.stem}")
    ax.set_xlabel("Time")
    ax.set_ylabel("GHI")

    return stats

def main_cluster(dir, n_clusters):
    climatology = monthly_mean(dir)
    random_state = 0
    kmeans = KMeans(n_clusters, init='k-means++', random_state=random_state)
    hierarchical = AgglomerativeClustering(n_clusters)
    engines = {kmeans: "K-Means", hierarchical:"Hierarchical"}

    fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": ccrs.PlateCarree()})
    for i, engine in enumerate(engines.keys()):
        unscaled_clusters, _ = cluster(climatology, engine)
        scaled_clusters, _ = cluster(climatology, engine, scale=True)

        plot_cluster(unscaled_clusters, axes[i, 0], n_clusters=n_clusters,
                     title=f"{engines[engine]} Clustering (unscaled)")
        plot_cluster(scaled_clusters, axes[i, 1], n_clusters=n_clusters,
                     title=f"{engines[engine]} Clustering (scaled)")

def main_summarise(dir):
     fig, axes = summary_visuals(dir)


def plot_cluster(cluster_da, ax=None, *, n_clusters, cmap="tab10", title=None):
    """Plot cluster labels on ax using one shared discrete colormap/norm,
    so cluster colours are consistent across every subplot that calls this.
    If ax is None, creates a new figure with a cartopy-projected axes (a
    plain matplotlib Axes, which .plot(ax=None, ...) would create instead,
    has no .coastlines())."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())

    base_cmap = plt.get_cmap(cmap, n_clusters)
    # BoundaryNorm carves the colormap into n_clusters discrete bins, with
    # edges at -0.5, 0.5, 1.5, ... so integer label k always lands in bin k.
    # Without it, .plot() would treat cluster ids as a continuous scale and
    # interpolate colours between them instead of giving each cluster a flat,
    # distinct colour.
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_clusters, 1), base_cmap.N)

    cluster_da.squeeze().plot(
        ax=ax,
        cmap=base_cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        add_colorbar=False,
    )
    format_labels(cluster_da, ax)
    if title:
        ax.set_title(title)
    ax.coastlines()
    handles = [Rectangle((0, 0), 1, 1, color=base_cmap(i)) for i in range(n_clusters)]
    ax.legend(handles, [f"Cluster {i}" for i in range(1,n_clusters+1)],
              loc="lower left", bbox_to_anchor=(1.02, 0))


def station_clusters(cluster_da: xr.DataArray, stations: list[dict]) -> dict[str, int]:
    """For each station, finds which cluster its nearest grid point in
    cluster_da (as returned by cluster()) belongs to.
    Returns a dict mapping station name -> cluster label (int).

    Example Usage:
    clusters_by_station = station_clusters(unscaled_clusters, STATIONS)"""
    return {
        station["name"]: int(cluster_da.sel(latitude=station["lat"], longitude=station["long"], method="nearest"))
        for station in stations
    }


def plot_cluster_ghi_with_stations(climatology: xr.DataArray, cluster_da: xr.DataArray, stations: list[dict], n_clusters: int, ncols: int = 3):
    """For each cluster 0..n_clusters-1, plots that cluster's average GHI
    (spatial mean over every location in the cluster, from climatology -
    the same array clustered to produce cluster_da) alongside individual GHI
    lines for each station whose nearest grid point falls in that cluster
    (see station_clusters), one subplot per cluster.

    Example Usage:
    fig, axes = plot_cluster_ghi_with_stations(climatology, unscaled_clusters, STATIONS, 6)"""
    clusters_by_station = station_clusters(cluster_da, stations)

    nrows = -(-n_clusters // ncols)  # ceil division
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flat

    # Cluster labels from sklearn are 0-indexed (0..n_clusters-1) - that's
    # what's actually stored in cluster_da/clusters_by_station, so data
    # operations below stay 0-indexed. Only the displayed title uses the
    # project's 1-indexed convention for clusters (k+1), matching plot_cluster.
    for k in range(n_clusters):
        ax = axes[k]
        cluster_avg = climatology.where(cluster_da == k).mean(dim=("latitude", "longitude"))
        cluster_avg.plot.line(ax=ax, color="black", linewidth=2, label="Cluster average")

        for station in stations:
            if clusters_by_station[station["name"]] != k:
                continue
            station_series = climatology.sel(latitude=station["lat"], longitude=station["long"], method="nearest")
            station_series.plot.line(ax=ax, label=station["name"])

        ax.set_title(f"Cluster {k + 1}")
        ax.set_ylabel("GHI")
        ax.legend(fontsize=8)

    for ax in axes[n_clusters:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def make_gif(input_path, output_path=None, cmap=GHI_CMAP, weekly=True, fps=4, **gif_kwargs):
    """
    Example Usage:
    make_gif(Path("aus_ghi/ghi_20210101_20211231.nc"))
    make_gif(input_path, Path("weekly.gif"), cmap=GHI_CMAP, fps=2)"""
    import geogif

    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path.with_name(f"{input_path.stem}.gif")

    with xr.open_dataarray(input_path, engine="netcdf4") as da:
        da = da.load()

    if weekly:
        da = da.resample(time="1W").mean()
        # geogif just rasters the array in index order (row 0 = top of the
        # image), with no awareness of coordinate direction. Our latitude
        # is ascending (index 0 = southernmost), so without reversing it
        # here the south ends up at the top of each frame - i.e. Australia
        # upside down.
        da = da.reindex(latitude=da.latitude[::-1])
        # geogif expects (time, y, x) dims for a single-band animation.
        da = da.rename({"latitude": "y", "longitude": "x"})

    geogif.gif(da, to=output_path, fps=fps, cmap=cmap, **gif_kwargs)
    return output_path


def monthly_mean(dir:Path):
    files = sorted(dir.glob("*.nc"))
    monthly_means = []
    years = []
    for file in files:
        with xr.open_dataarray(file, engine="netcdf4") as da:
            years.append(int(da.time.dt.year.values[0]))
            monthly_means.append(da.groupby("time.month").mean(dim="time").load())

    monthly_means = xr.concat(monthly_means, dim="year", coords="minimal")
    monthly_means = monthly_means.assign_coords(year=years)
    return monthly_means.mean(dim="year")


def summary_visuals(dir):
    dir = Path(dir)
    time_means = []
    spatial_vars = []
    spatial_means = []
    files = sorted(dir.glob("*.nc"))
    if not files:
        raise RuntimeError(f"No .nc files found in {dir}")

    for file in files:
        with xr.open_dataarray(file, engine="netcdf4") as da:
            # Mean across all time for each location
            time_means.append(da.mean(dim="time").load())

            # Variance across all locations for each time
            spatial_vars.append(da.var(dim=("latitude", "longitude")).load())

            # Mean GHI across entire region
            spatial_means.append(da.mean(dim=("latitude", "longitude")).load())

    # Aggregate over all years
    mean_ghi_5yr = xr.concat(time_means, dim="file").mean(dim="file")
    # Mean each year's variance series first, then average across years,
    # so a leap year's extra day can't skew a naive elementwise mean.
    spatial_variance_5yr = float(np.mean([sv.mean().item() for sv in spatial_vars]))

    for da in spatial_means:
        da.coords["dayofyear"] = da["time"].dt.dayofyear
    avg_daily_ghi = (
            xr.concat(spatial_means, dim="time")
            .groupby("dayofyear")
            .mean(dim="time")
        )

    # plt.subplots(subplot_kw=...) forces the same projection on every axes,
    # so a mixed-axes figure (map + line plot) needs fig.add_subplot per axes
    # instead, passing projection= only to the one that needs to be a GeoAxes.
    fig = plt.figure(figsize=(14, 5))
    map_ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    line_ax = fig.add_subplot(1, 2, 2)

    plot_ghi(mean_ghi_5yr, ax=map_ax, extent=AUSTRALIA_EXTENT)
    format_labels(mean_ghi_5yr, map_ax)
    map_ax.set_title("Mean GHI by Location")

    avg_daily_ghi.plot.line(ax=line_ax)
    line_ax.set_xlabel("Day of Year")
    line_ax.set_ylabel("GHI")
    line_ax.set_title("Average Daily GHI Across the Year")

    fig.suptitle("GHI Summary")
    fig.subplots_adjust(wspace=0.45)
    fig.text(
        0.5, 0.3,
        f"Average spatial\nvariance: {spatial_variance_5yr:.2f}",
        ha="center", va="center", fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    return fig, (map_ax, line_ax)


SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
SEASON_LABELS = {
    "DJF": "Summer",
    "MAM": "Autumn",
    "JJA": "Winter",
    "SON": "Spring",
}

def seasonal_climatology(yearly: list):
    """
    Returns (seasonal_mean, seasonal_std), each a DataArray with a "season"
    dim (ordered DJF, MAM, JJA, SON) plus latitude/longitude."""


    full = xr.concat(yearly, dim="time", coords="minimal").sortby("time")
    grouped = full.groupby("time.season")
    seasonal_mean = grouped.mean(dim="time").reindex(season=SEASON_ORDER)
    seasonal_std = grouped.std(dim="time").reindex(season=SEASON_ORDER)
    return seasonal_mean, seasonal_std


def plot_seasonal_ghi(seasonal_da, title=None, cmap=GHI_CMAP, **plot_kwargs):
    """Plots a DataArray with a "season" dim (values DJF/MAM/JJA/SON, as
    returned by seasonal_climatology) as a 2x2 grid of maps, one per season
    (summer, autumn, winter, spring), using plot_ghi/format_labels.

    Example Usage:
    seasonal_mean, seasonal_std = seasonal_climatology(Path("aus_ghi"))
    plot_seasonal_ghi(seasonal_mean, title="Seasonal Average GHI (5-Year Mean)")
    plot_seasonal_ghi(seasonal_std, title="Seasonal GHI Std Dev (5-Year Mean)")"""
    fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 10))
    for ax, season in zip(axes.flat, SEASON_ORDER):
        da = seasonal_da.sel(season=season)
        plot_ghi(da, ax=ax, extent=AUSTRALIA_EXTENT, cmap=cmap, **plot_kwargs)
        ax.set_title(SEASON_LABELS[season])
        format_labels(da, ax)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def to_feature_matrix(da: xr.DataArray):
    """Converts a (latitude, longitude, ...) DataArray - e.g. a climatology
    from monthly_mean - into a 2D sklearn-ready feature matrix X
    (locations x features), stacking latitude/longitude into one "location"
    dim and dropping any location with a NaN in any feature.
    Returns (X, stacked, valid) - stacked is the intermediate
    location-stacked DataArray and valid is the boolean mask of which
    locations survived the NaN drop, both needed to unstack cluster labels
    back onto the map afterward (see cluster()).

    Example Usage:
    X, stacked, valid = to_feature_matrix(climatology)"""
    stacked = da.stack(location=("latitude", "longitude")).transpose("location", ...)
    features = stacked.values
    valid = ~np.isnan(features).any(axis=1)
    X = features[valid]
    return X, stacked, valid


def cluster(da, engine, scale=False):
    X, stacked, valid = to_feature_matrix(da)
    if scale:
        X = StandardScaler().fit_transform(X)

    fit = engine.fit(X)

    label_array = np.full(stacked.sizes["location"], np.nan)
    label_array[valid] = fit.labels_
    cluster_da = xr.DataArray(
    label_array, coords={"location": stacked.location}, dims="location"
    ).unstack("location")

    return cluster_da, fit


def elbow_silhouette_analysis(X, cluster_range=range(2, 11), n_init=3):
    """Runs KMeans over each k in cluster_range on the given feature matrix
    X, computing inertia (for the elbow method) and mean silhouette score
    for each k. To reduce the effect of any single random_state's luck, each
    k is fit n_init separate times (random_state 0..n_init-1) and the
    inertia/silhouette are averaged across those runs - distinct from
    KMeans' own internal n_init="auto", which instead keeps only the single
    best-inertia run out of several initialisations under one random_state;
    both are used together here (n_init="auto" per fit, plus this outer
    n_init-run average across fits).
    Works on any X - not specific to GHI data - so this can be reused for
    any clustering analysis, not just the aus_ghi climatology.
    Returns (elbow_fig, silhouette_fig).

    Example Usage:
    elbow_fig, silhouette_fig = elbow_silhouette_analysis(X)
    elbow_fig, silhouette_fig = elbow_silhouette_analysis(X, range(3, 13), n_init=5)"""
    cluster_range = list(cluster_range)
    mean_inertias = []
    mean_silhouettes = []

    for k in cluster_range:
        inertias = []
        silhouettes = []
        for seed in range(n_init):
            model = KMeans(n_clusters=k, init="k-means++", random_state=seed, n_init="auto").fit(X)
            inertias.append(model.inertia_)
            silhouettes.append(silhouette_score(X, model.labels_))
        mean_inertias.append(np.mean(inertias))
        mean_silhouettes.append(np.mean(silhouettes))

    elbow_fig, elbow_ax = plt.subplots()
    elbow_ax.plot(cluster_range, mean_inertias, "o-")
    elbow_ax.set_xlabel("Number of clusters")
    elbow_ax.set_ylabel(f"Inertia (mean of {n_init} runs)")
    elbow_ax.set_title("Elbow Method")

    silhouette_fig, silhouette_ax = plt.subplots()
    silhouette_ax.plot(cluster_range, mean_silhouettes, "o-")
    silhouette_ax.set_xlabel("Number of clusters")
    silhouette_ax.set_ylabel(f"Silhouette Score (mean of {n_init} runs)")
    silhouette_ax.set_title("Silhouette Analysis")

    return elbow_fig, silhouette_fig



if __name__ == "__main__":
    main()
