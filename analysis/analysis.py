from dataclasses import dataclass, field
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .clustering import SEASON_ORDER

DAYS_PER_WEEK = 7
PERIODS = {"MJJAS": [5, 6, 7, 8, 9], "NDJFM": [11, 12, 1, 2, 3]}
SEASONS = ["DJF", "MAM", "JJA", "SON"]


@dataclass
class DataConfig:
    """Configuration for the private data-preparation pipeline."""

    regrid_equal_area: bool = True
    aggregate_years: bool = False
    season: Literal["DJF", "MAM", "JJA", "SON"] | None = None
    scale: bool = False
    representation: Literal["daily", "weekly", "weekly_mean", "weekly_sequence"] = "daily"
    overlap_sequences: bool = True
    seq_len: int = 7
    pca: bool = False
    pca_threshold: float | None = None
    pca_max_ncomp: int = 100


@dataclass
class FeatureData:
    """Validated spatial data and the metadata required to build features."""

    data: xr.DataArray
    valid_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    grid_type: Literal["equal_area", "lat_lon"] = field(init=False)
    spatial_dims: tuple[str, str] = field(init=False)

    def __post_init__(self) -> None:
        if "latitude" not in self.data.coords or "longitude" not in self.data.coords:
            raise ValueError("FeatureData requires latitude and longitude coordinates")
        if {"x", "y"}.issubset(self.data.dims):
            self.grid_type = "equal_area"
            self.spatial_dims = ("y", "x")
        elif {"latitude", "longitude"}.issubset(self.data.dims):
            self.grid_type = "lat_lon"
            self.spatial_dims = ("latitude", "longitude")
        else:
            raise ValueError("FeatureData requires x/y or latitude/longitude dimensions")
        if self.valid_mask is None:
            non_spatial_dims = [dim for dim in self.data.dims if dim not in self.spatial_dims]
            self.valid_mask = ~np.isnan(self.data).any(dim=non_spatial_dims).values

    def __repr__(self) -> str:
        valid = int(np.asarray(self.valid_mask).sum()) if self.valid_mask is not None else 0
        coords = ", ".join(self.data.coords)
        return (
            f"FeatureData(size={self.data.size}, dims={self.data.dims}, "
            f"shape={self.data.shape}, grid_type={self.grid_type!r}, "
            f"valid_points={valid}, coordinates=[{coords}])"
        )


class AnalysisBase:
    """Reusable preparation pipeline shared by machine-learning techniques."""

    def __init__(self, data: Any, config: DataConfig | None = None) -> None:
        self.config = config or DataConfig()
        self.data = self._coerce_data(data)
        if self.config.regrid_equal_area and not {"x", "y"}.issubset(self.data.data.dims):
            warnings.warn(
                "Analysis data is not indexed by x/y; continuing without regridding. "
                "Use DataLoader's default regrid_equal_area=True for the expected grid.",
                UserWarning,
                stacklevel=2,
            )
        self.feature_data: FeatureData | None = None
        self.feature_matrix: np.ndarray | None = None
        self.warnings: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.prepare()
        self._record_warnings(caught)

    def _record_warnings(self, caught: list[warnings.WarningMessage]) -> None:
        self.warnings.extend(str(item.message) for item in caught)

    @staticmethod
    def _coerce_data(data: Any) -> FeatureData:
        if isinstance(data, FeatureData):
            return data
        if hasattr(data, "data") and isinstance(data.data, xr.DataArray):
            return FeatureData(data.data)
        if isinstance(data, xr.DataArray):
            return FeatureData(data)
        raise TypeError("data must be a FeatureData, DataLoader, or xarray.DataArray")

    def prepare(self) -> np.ndarray:
        """Run the configured preparation pipeline and return its feature matrix."""
        steps = [self._select_season]
        if self.config.aggregate_years:
            steps.append(self._aggregate_years)
        steps.extend([self._extract_representation, self._compute_valid_mask])
        if self.config.scale:
            steps.append(self._scale)

        for step in steps:
            self.data = step(self.data)
        self.feature_data = self.data
        self.feature_matrix = self._to_feature_matrix_dynamic(self.data)
        return self.feature_matrix

    def _select_season(self, data: FeatureData) -> FeatureData:
        if self.config.season is None:
            return data
        if "time" not in data.data.coords:
            return data
        selected = data.data.where(data.data.time.dt.season == self.config.season, drop=True)
        return FeatureData(selected, metadata=data.metadata)

    def _aggregate_years(self, data: FeatureData) -> FeatureData:
        if "time" not in data.data.dims:
            return data
        aggregated = data.data.groupby("time.dayofyear").mean(dim="time")
        return FeatureData(aggregated, metadata=data.metadata)

    def _extract_representation(self, data: FeatureData) -> FeatureData:
        if self.config.representation == "daily":
            return data
        if self.config.representation not in {"weekly", "weekly_mean", "weekly_sequence"}:
            raise ValueError(f"Unsupported representation: {self.config.representation!r}")
        sample_dim = data.data.dims[0]
        sequence_len = DAYS_PER_WEEK if self.config.representation in {"weekly", "weekly_mean"} else self.config.seq_len
        if sequence_len <= 0:
            raise ValueError("seq_len must be a positive integer")

        n_samples = data.data.sizes[sample_dim]
        is_sequence = self.config.representation == "weekly_sequence"
        if is_sequence and self.config.overlap_sequences:
            n_sequences = n_samples - sequence_len + 1
            starts = np.arange(max(n_sequences, 0))
        else:
            n_sequences = n_samples // sequence_len
            starts = np.arange(n_sequences) * sequence_len
        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for sequence length {sequence_len}"
            )

        values = np.stack([
            data.data.isel({sample_dim: slice(start, start + sequence_len)}).values
            for start in starts
        ])
        coords = {name: value for name, value in data.data.coords.items() if name != sample_dim}
        coords["sample"] = data.data[sample_dim].isel({sample_dim: starts}).values
        if self.config.representation in {"weekly", "weekly_mean"}:
            values = values.mean(axis=1)
            dims = ("sample", *data.data.dims[1:])
        else:
            dims = ("sample", "sequence_step", *data.data.dims[1:])
            coords["sequence_step"] = np.arange(sequence_len)
        result = xr.DataArray(values, dims=dims, coords=coords, attrs=data.data.attrs)
        return FeatureData(result, metadata={
            **data.metadata,
            "samples_per_instance": sequence_len,
            "overlap_sequences": self.config.overlap_sequences if is_sequence else False,
        })

    def _compute_valid_mask(self, data: FeatureData) -> FeatureData:
        non_spatial_dims = [dim for dim in data.data.dims if dim not in data.spatial_dims]
        valid = ~np.isnan(data.data).any(dim=non_spatial_dims).values
        return FeatureData(data.data, valid, data.metadata)

    def _scale(self, data: FeatureData) -> FeatureData:
        sample_dim = data.data.dims[0]
        mean = data.data.mean(dim=sample_dim)
        std = data.data.std(dim=sample_dim).where(lambda value: value > 0)
        return FeatureData((data.data - mean) / std, metadata={**data.metadata, "scaled": True})

    def _to_feature_matrix_dynamic(self, data: FeatureData) -> np.ndarray:
        values = data.data.values
        valid = np.asarray(data.valid_mask)
        flat = values.reshape(values.shape[0], -1)
        return flat[:, np.broadcast_to(valid, values.shape[1:]).reshape(-1)]

    def _unflatten_to_grid(self, vector: np.ndarray) -> xr.DataArray:
        data = self.feature_data
        if data is None or data.valid_mask is None:
            raise RuntimeError("prepare() must be called before reconstructing a grid")
        full = np.full(data.valid_mask.shape, np.nan)
        full[data.valid_mask] = vector
        spatial = data.spatial_dims
        coords = {dim: data.data.coords[dim] for dim in spatial}
        for name in ("latitude", "longitude"):
            if name in data.data.coords:
                coords[name] = data.data.coords[name]
        return xr.DataArray(full, dims=spatial, coords=coords)


def dayofyear_to_season(dayofyear: np.ndarray) -> np.ndarray:
    """Maps 1-based day-of-year values to their Southern Hemisphere
    meteorological season (DJF/MAM/JJA/SON, see clustering.SEASON_ORDER),
    via a fixed non-leap reference year (2001). Safe here because
    build_weekly_arrays never keeps day 365/366 (52 weeks * 7 days = 364
    <= 365), and a given day-of-year falls in the same season regardless of
    which calendar year/leap status it came from."""
    reference_dates = pd.Timestamp("2001-01-01") + pd.to_timedelta(np.asarray(dayofyear) - 1, unit="D")
    season = np.array([_MONTH_TO_SEASON[m] for m in reference_dates.month])
    assert set(season) == set(SEASON_ORDER), "season mapping produced unexpected labels"
    return season

_MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

def assign_season_instance(clim: xr.DataArray) -> np.ndarray:
    """Labels each sample along clim's leading dimension with its Southern
    Hemisphere season *instance* - e.g. "DJF-2022" for the Dec 2021-Feb 2022
    summer - rather than just its season *type* ("DJF"), so
    seasonal_standardize can normalize each day against its own season
    occurrence's mean/std instead of lumping every year's occurrence of that
    season together.

    Handles both of load_climatology's outputs:
      - average_years=True (leading dim "dayofyear", no year information):
        every season type occurs exactly once in the single climatological
        year, so its "instance" is just that season type - dayofyear_to_season
        unchanged.
      - average_years=False (leading dim "time", real multi-year dates):
        season type is read off the real calendar month, and December is
        folded into the *following* January/February's instance (Southern
        Hemisphere summer spans a year boundary - Dec 2021 + Jan/Feb 2022 is
        one "DJF-2022" instance, not two separate groups). A season instance
        at the very start or end of the data range may end up with fewer
        days than the others (e.g. if the record starts in January, that
        first DJF instance is missing its December) - seasonal_standardize
        computes each instance's own mean/std from however many days it
        actually has, so this is fine, just occasionally a smaller sample."""
    sample_dim = clim.dims[0]
    if sample_dim == "dayofyear":
        return dayofyear_to_season(clim["dayofyear"].values)

    if sample_dim != "time":
        raise ValueError(f"assign_season_instance: unexpected leading dim {sample_dim!r}")

    month = clim["time"].dt.month.values
    year = clim["time"].dt.year.values
    season = np.array([_MONTH_TO_SEASON[m] for m in month])
    assert set(season) <= set(SEASON_ORDER), "season mapping produced unexpected labels"
    season_year = np.where(month == 12, year + 1, year)
    return np.array([f"{s}-{y}" for s, y in zip(season, season_year)])

def seasonal_standardize(clim: xr.DataArray) -> xr.DataArray:
    """Per-location seasonal z-score: for each season *instance* (see
    assign_season_instance) and each location, subtracts that instance's
    mean and divides by its std dev - both computed across just that
    instance's days.

    Example Usage:
    clim = seasonal_standardize(load_climatology())
    clim = seasonal_standardize(load_climatology(average_years=False))"""
    sample_dim = "time"
    instance = assign_season_instance(clim)
    clim = clim.assign_coords(season_instance=(sample_dim, instance))
    grouped = clim.groupby("season_instance")

    season_mean = grouped.mean(dim=sample_dim)
    season_std = grouped.std(dim=sample_dim)
    has_zero_std = bool((season_std.notnull() & (season_std == 0)).any().item())
    if has_zero_std:
        print(
            "Warning: seasonal_standardize: some location/season-instance combinations "
            "have zero std dev (constant GHI all season)"
        )
    season_std = season_std.where(season_std > 0)

    anomaly = clim.groupby("season_instance") - season_mean
    return anomaly.groupby("season_instance") / season_std

def build_weekly_arrays(
    clim: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (weekly_mean, weekly_sequence, valid, latitude, longitude):
      - weekly_mean: ndarray (week, latitude, longitude) - one mean
        snapshot per week
      - weekly_sequence: ndarray (week, day_in_week, latitude, longitude) -
        full 7-day resolution per week
      - valid: boolean (latitude, longitude) mask of locations with GHI data
        in every single week/day of `clim`
      - latitude, longitude: coordinate arrays for unflatten_to_grid"""
    sample_dim = clim.dims[0]
    n_days = clim.sizes[sample_dim]
    n_weeks = n_days // DAYS_PER_WEEK
    dropped = n_days - n_weeks * DAYS_PER_WEEK
    if dropped:
        print(
            f"build_weekly_arrays: dropping last {dropped} sample(s) of the "
            f"{n_days}-sample {sample_dim} series to keep {n_weeks} complete "
            f"{DAYS_PER_WEEK}-day weeks"
        )

    trimmed = clim.isel({sample_dim: slice(0, n_weeks * DAYS_PER_WEEK)})
    if "latitude" in clim.dims and "longitude" in clim.dims:
        y, x = trimmed.latitude.values, trimmed.longitude.values
    elif "x" in clim.dims and "y" in clim.dims:
        assert "latitude" in clim.coords and "longitude" in clim.coords, "Climate data array does not have lat, lon coordinates."
        x, y = trimmed.x.values, trimmed.y.values
    else:
        raise ValueError("Data array has invalid coordinates")

    arr = trimmed.values.reshape(n_weeks, DAYS_PER_WEEK, len(y), len(x))
    valid = ~np.isnan(arr).any(axis=(0, 1))
    intermittent = (~np.isnan(arr[0, 0]) & ~valid).sum()
    if intermittent:
        print(
            f"build_weekly_arrays: {intermittent} location(s) are NaN in "
            f"only some weeks/days (not the usual always-missing ocean "
            f"cells) - dropping those too, keeping only locations with data "
            f"in every single week/day"
        )

    weekly_mean = arr.mean(axis=1)
    return weekly_mean, arr, valid, y, x

def build_daily_arrays(
    clim: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like build_weekly_arrays, but keeps every sample of clim (as returned
    by load_climatology, either the single climatological year or - with
    average_years=False - every day of every year) as its own sample instead
    of chunking into 7-day weeks - no trimming is needed here since there's
    no week-length divisibility constraint to satisfy at daily resolution.

    Returns (daily_values, valid, latitude, longitude):
      - daily_values: ndarray (dayofyear or time, latitude, longitude) - one
        snapshot per day (== clim.values)
      - valid: boolean (latitude, longitude) mask of locations with GHI data
        on every single day of `clim` (usually just the ocean/no-coverage
        mask, which is constant across time - see module docstring - but
        seasonal_standardize's small boundary season instances, see
        assign_season_instance, can occasionally zero out a location's std
        dev and turn it to NaN in just that instance, so this is computed as
        the intersection of "not NaN" across every day rather than assumed
        constant)
      - latitude, longitude: coordinate arrays for unflatten_to_grid"""
    latitude, longitude = clim.latitude.values, clim.longitude.values
    arr = clim.values

    valid = ~np.isnan(arr).any(axis=0)
    intermittent = (~np.isnan(arr[0]) & ~valid).sum()
    if intermittent:
        print(
            f"build_daily_arrays: {intermittent} location(s) are NaN on "
            f"only some days (not the usual always-missing ocean cells) - "
            f"dropping those too, keeping only locations with data on every "
            f"single day"
        )

    return arr, valid, latitude, longitude

def to_feature_matrix(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Flattens every non-week axis of `values` (week, ...) into one feature
    vector per week, keeping only locations flagged by `valid`.

    Example Usage:
    X1 = to_feature_matrix(weekly_mean, valid)          # (n_weeks, n_valid)
    X2 = to_feature_matrix(weekly_sequence, valid)      # (n_weeks, 7 * n_valid)"""
    n_weeks = values.shape[0]
    flat = values.reshape(n_weeks, -1)
    valid_flat = np.broadcast_to(valid, values.shape[1:]).reshape(-1)
    return flat[:, valid_flat]

def unflatten_to_grid(
    vector: np.ndarray,
    valid: np.ndarray,
    latitude: Any,
    longitude: Any,
) -> xr.DataArray:
    """Inverse of to_feature_matrix for a single (n_valid,) vector: scatters
    it back onto the full (latitude, longitude) grid, filling every dropped
    location with NaN so it plots (and coastlines) exactly like the source
    data."""
    valid = np.asarray(valid, dtype=bool)
    vector = np.asarray(vector)
    expected = int(valid.sum())
    if vector.size != expected:
        raise ValueError(
            f"unflatten_to_grid expected {expected} values for the valid mask, "
            f"got {vector.size}"
        )
    full = np.full(valid.shape, np.nan, dtype=np.result_type(vector.dtype, float))
    full[valid] = vector.reshape(-1)
    if isinstance(latitude, xr.DataArray) and isinstance(longitude, xr.DataArray):
        if latitude.ndim == longitude.ndim == 1:
            dims = ("latitude", "longitude")
        elif latitude.ndim == longitude.ndim == 2 and latitude.dims == longitude.dims:
            dims = latitude.dims
        else:
            raise ValueError("latitude and longitude must both be 1-D or matching 2-D arrays")
        coords = {"latitude": latitude, "longitude": longitude}
    elif np.ndim(latitude) == 2 and np.ndim(longitude) == 2:
        dims = ("y", "x")
        coords = {
            "latitude": (dims, latitude),
            "longitude": (dims, longitude),
        }
    else:
        dims = ("latitude", "longitude")
        coords = {"latitude": latitude, "longitude": longitude}
    return xr.DataArray(full, coords=coords, dims=dims)
