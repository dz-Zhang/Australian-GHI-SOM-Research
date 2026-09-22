import xarray as xr
import numpy as np
from .plotting import AUSTRALIA_EXTENT
from datetime import date, datetime, time, timedelta
from pathlib import Path
from functools import partial
from contextlib import contextmanager
import tempfile
from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass, field
from pyproj import Transformer

BASE_URL_TEMPLATE = (
    "https://thredds.nci.org.au/thredds/dodsC/"
    "rv74/satellite-products/arc/der/himawari-ahi/solar/"
    "{product}/latest"
)

OPCODES = {
    "d": "IDE02326",
    "h": "IDE02327",
    "s": "IDE00326",
}

VARIABLES = {
    "d": "daily_integral_of_surface_global_irradiance",
    "h": "hourly_integral_of_surface_global_irradiance",
    "s": "surface_global_irradiance",
}

InterpMethod = Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic", "quintic", "polynomial", "pchip", "barycentric", "krogh", "akima", "makima"]

COARSEN = 23 # roughly 50km by 50km regions

# A "day" of p1h/p1s data doesn't span 00:00-23:59 of that calendar date
WINDOW_START = time(18, 30)
WINDOW_END = {
    "h": time(10, 30),
    "s": time(11, 30),
}
WINDOW_STEP = {
    "h": timedelta(hours=1),
    "s": timedelta(minutes=10),
}
SYDNEY_EXTENT = (150.55, 151.45, -34.25, -33.35) #long min/max, lat min/max

GROUP_SIZE = 10  # thredds.nci.org.au starts raising I/O errors somewhere
                 # around 20 simultaneously-open connections

@dataclass
class DataLoader:
    """Loader and persistence interface for gridded irradiance data."""

    data: xr.DataArray
    missing_ranges: list[tuple[date, date]] = field(default_factory=list)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        regrid_equal_area: bool = True,
        resolution_km: float = 55,
        regrid_method: InterpMethod = "linear",
    ) -> "DataLoader":
        """Load one NetCDF file or concatenate all ``*.nc`` files in a folder."""
        source = Path(path)
        files = [source] if source.is_file() else sorted(source.glob("*.nc"))
        if not files:
            raise RuntimeError(f"No .nc files found in {source}")

        arrays: list[xr.DataArray] = []
        progress = tqdm(files, desc="Loading NetCDF files", unit="file")
        for file_path in progress:
            with xr.open_dataarray(file_path, engine="netcdf4") as da:
                arrays.append(da.load())
        data = arrays[0] if len(arrays) == 1 else xr.concat(arrays, dim="time", coords="minimal")
        data = data.sortby("time") if "time" in data.dims else data
        if regrid_equal_area:
            data = regrid_to_equal_area(data, resolution_km, regrid_method)
        return cls(data=data)

    @classmethod
    def from_thredds(
        cls,
        start: date,
        end: date,
        *,
        time_res: Literal["d", "h", "s"] = "d",
        extent: tuple[float, float, float, float] = SYDNEY_EXTENT,
        coarsen: int | None = None,
        batch_days: int = 30,
        variable: str | None = None,
        regrid_equal_area: bool = True,
        resolution_km: float = 55,
        regrid_method: InterpMethod = "linear",
    ) -> "DataLoader":
        """Fetch data from the THREDDS server and return a loader instance."""
        data = open_ghi(start, end, time_res, extent, coarsen, batch_days, variable)
        if regrid_equal_area:
            data = regrid_to_equal_area(data, resolution_km, regrid_method)
        return cls(data=data)

    def write(self, output_path: str | Path) -> Path:
        """Write the loaded data to a NetCDF file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_netcdf(output)
        return output

    def climatology(self, average_years: bool = False) -> xr.DataArray:
        """Return the loaded data, optionally averaged by day of year."""
        if not average_years:
            return self.data
        if "time" not in self.data.dims:
            raise ValueError("Cannot average years without a time dimension")
        return self.data.groupby("time.dayofyear").mean(dim="time")


def iter_days(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)

def iter_window(start: date, end: date, time_res: str):
    """Yields each file timestamp for every nominal day in [start, end],
    covering that day's daylight window: WINDOW_START (previous day) through
    WINDOW_END[time_res] (current day), stepping by WINDOW_STEP[time_res]."""
    step = WINDOW_STEP[time_res]
    window_end = WINDOW_END[time_res]
    day = start
    while day <= end:
        current = datetime.combine(day - timedelta(days=1), WINDOW_START)
        stop = datetime.combine(day, window_end)
        while current <= stop:
            yield current
            current += step
        day += timedelta(days=1)

def iter_dates(start: date, end: date, time_res: str):
    match time_res:
        case "d":
            yield from iter_days(start, end)
        case "h" | "s":
            yield from iter_window(start, end, time_res)
        case _:
            raise ValueError(f"Unsupported time_res: {time_res!r}")

def build_url(when: date | datetime, time_res: str) -> str:
    match time_res:
        case "d":
            folder_date = when
            timestamp = f"{when:%Y%m%d}0000"
        case "h" | "s":
            folder_date = when.date() if when.time() < WINDOW_START else when.date() + timedelta(days=1)
            timestamp = f"{when:%Y%m%d%H%M}"
        case _:
            raise ValueError(f"Unsupported time_res: {time_res!r}")

    opcode = OPCODES[time_res]
    base_url = BASE_URL_TEMPLATE.format(product=f"p1{time_res}")

    return (
        f"{base_url}/{folder_date:%Y/%m/%d}/"
        f"{opcode}.{timestamp}.nc"
    )

def _crop_to_variable(ds: xr.Dataset, variable: str, extent) -> xr.Dataset:
    """Keep the requested variable and crop it to ``extent`` locally."""
    if variable not in ds:
        raise KeyError(
            f"{variable!r} is not present. "
            f"Available variables: {list(ds.data_vars)}"
        )

    return ds[[variable]].sel(
        latitude=slice(*extent[2:]),
        longitude=slice(*extent[:2]),
    )

def _crop_and_coarsen(ds: xr.Dataset, variable: str, extent, coarsen) -> xr.Dataset:
    """preprocess callback for open_mfdataset: crop (see _crop_to_variable),
    then - if coarsen is not None - spatially coarsen by averaging lat/long
    blocks of size coarsen x coarsen. coarsen only touches the lat/long dims,
    so doing it per-file (each file being one time step, full spatial grid)
    before the per-file data is ever loaded is equivalent to doing it once
    after concatenating over time, but means only the small coarsened array
    ever gets materialized - not the full-resolution crop first."""
    cropped = _crop_to_variable(ds, variable, extent)
    if coarsen is not None:
        cropped = cropped.coarsen(latitude=coarsen, longitude=coarsen, boundary="trim").mean()
    return cropped

def _split_date_range(start: date, end: date, batch_days: int):
    """Yields (batch_start, batch_end) sub-ranges covering [start, end], each
    at most batch_days long."""
    batch_start = start
    while batch_start <= end:
        batch_end = min(batch_start + timedelta(days=batch_days - 1), end)
        yield batch_start, batch_end
        batch_start = batch_end + timedelta(days=1)

def _fetch_one(when: datetime | date, time_res: str, extent, variable: str, coarsen):
    """Opens, crops, coarsens (if coarsen is not None), and loads a single
    file - coarsen is applied before .load() so only the small coarsened
    array is ever materialized, not the full-resolution crop.
    Returns None (and logs) if the file can't be opened or its time
    coordinate didn't decode properly."""
    url = build_url(when, time_res)
    try:
        with xr.open_dataset(url, engine="netcdf4") as ds:
            cropped = _crop_and_coarsen(ds, variable, extent, coarsen)
            cropped.load()
    except OSError as e:
        tqdm.write(f"{e} : {when}")
        return None

    if not np.issubdtype(cropped["time"].dtype, np.datetime64):
        # Some source files are missing CF time units/calendar attrs, so
        # xarray can't decode "time" into datetime64 and leaves it as a raw
        # int - concatenating that with properly-decoded chunks later raises
        # a DTypePromotionError. Treat it the same as an unreadable file
        # rather than letting it crash the whole batch.
        tqdm.write(f"Malformed time coordinate (dtype={cropped['time'].dtype}), skipping : {when}")
        return None

    return cropped

def _fetch_group(group: list, time_res: str, extent, variable: str, coarsen):
    """Fetches a small group of files (<= GROUP_SIZE) in one open_mfdataset
    call instead of one xr.open_dataset call per file. Crop and coarsen (if
    coarsen is not None) both happen via preprocess, before open_mfdataset's
    .load() - so only the small coarsened per-file arrays get materialized,
    not the full-resolution crop first.
    Falls back to fetching the group one file at a time if the group-level
    open fails (e.g. the server briefly refuses one of the connections), so
    one bad group doesn't lose the whole group's data.
    Returns None if nothing in the group was accessible."""
    urls = [build_url(when, time_res) for when in group]
    preprocess = partial(_crop_and_coarsen, variable=variable, extent=extent, coarsen=coarsen)

    try:
        with xr.open_mfdataset(
            urls,
            engine="netcdf4",
            preprocess=preprocess,
            data_vars="all",
            coords="minimal",
            compat="override",
            combine="by_coords",
            combine_attrs="override",
            parallel=False,
        ) as group_ds:
            group_ds = group_ds.load()
    except Exception as e:
        # Not just OSError: a single malformed file mixed into an otherwise
        # good group (e.g. undecoded int time coordinate - see _fetch_one)
        # makes open_mfdataset raise things like DTypePromotionError or
        # MergeError from inside the combine step, which aren't OSErrors.
        # Whatever the cause, fall back to fetching the group one file at a
        # time so a single bad file doesn't lose the whole group.
        tqdm.write(f"{e} : group {group[0]:%Y-%m-%d %H:%M}-{group[-1]:%Y-%m-%d %H:%M}, falling back to per-file fetch")
        chunks = [c for c in (_fetch_one(when, time_res, extent, variable, coarsen) for when in group) if c is not None]
        if not chunks:
            return None
        return xr.concat(chunks, dim="time")

    if not np.issubdtype(group_ds["time"].dtype, np.datetime64):
        tqdm.write(f"Malformed time coordinate in group {group[0]:%Y-%m-%d %H:%M}-{group[-1]:%Y-%m-%d %H:%M}, skipping")
        return None

    return group_ds

def _fetch_batch(batch_start: date, batch_end: date, time_res: str, extent, coarsen, variable: str):
    """Fetches, crops, and (if coarsen is not None) coarsens every file for
    [batch_start, batch_end], GROUP_SIZE files at a time via open_mfdataset -
    coarsen happens per-group/per-file, before each piece is loaded (see
    _fetch_group/_fetch_one), so this only ever holds the small coarsened
    data in memory, not the full-resolution crop.
    Returns None if nothing in the batch was accessible."""
    dates = list(iter_dates(batch_start, batch_end, time_res))
    groups = [dates[i:i + GROUP_SIZE] for i in range(0, len(dates), GROUP_SIZE)]

    chunks = []
    progress = tqdm(groups, desc=f"Fetching {batch_start:%Y-%m-%d}", unit="group", leave=False)
    for group in progress:
        progress.set_postfix(date=f"{group[0]:%Y-%m-%d %H:%M}")
        try:
            group_ds = _fetch_group(group, time_res, extent, variable, coarsen)
        except Exception as e:
            # _fetch_group already has its own fallback for group-level
            # failures; this is a last-resort net so a single group's
            # unexpected failure can't take down the whole batch (and, via
            # _fetch_in_batches, the whole multi-batch run).
            tqdm.write(f"{e} : group {group[0]:%Y-%m-%d %H:%M}-{group[-1]:%Y-%m-%d %H:%M}, skipping")
            group_ds = None
        if group_ds is not None:
            chunks.append(group_ds)

    if not chunks:
        return None

    return xr.concat(chunks, dim="time").drop_duplicates(dim="time").sortby("time")

def _fetch_in_batches(start: date, end: date, time_res: str, extent, coarsen, variable: str, batch_dir: Path, batch_days: int) -> tuple[list[Path], list[tuple[date, date]]]:
    """Fetches [start, end] in batch_days-sized batches, writing each batch to
    its own file under batch_dir as soon as it's ready (never appending to an
    existing file - to_netcdf(mode="a") silently overwrites rather than truly
    appending, so each batch gets a distinct path instead). This bounds peak
    memory to one batch at a time rather than the whole range.
    A batch that fails entirely (nothing fetched, or the write itself fails)
    is recorded as missing rather than aborting the rest of the run - one bad
    batch shouldn't cost you every other batch that did succeed.
    Returns (written batch file paths, list of (batch_start, batch_end) ranges
    that came back empty or failed to write)."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_paths = []
    missing_ranges = []
    for i, (batch_start, batch_end) in enumerate(tqdm(list(_split_date_range(start, end, batch_days)), desc="Batches", unit="batch")):
        try:
            batch = _fetch_batch(batch_start, batch_end, time_res, extent, coarsen, variable)
        except Exception as e:
            tqdm.write(f"{e} : batch {batch_start}-{batch_end}, skipping")
            missing_ranges.append((batch_start, batch_end))
            continue

        if batch is None:
            missing_ranges.append((batch_start, batch_end))
            continue

        batch_path = batch_dir / f"batch_{i:04d}.nc"
        try:
            batch.to_netcdf(batch_path)
        except Exception as e:
            tqdm.write(f"{e} : failed writing batch {batch_start}-{batch_end}, skipping")
            missing_ranges.append((batch_start, batch_end))
            continue
        finally:
            batch.close()
            del batch
        batch_paths.append(batch_path)
    return batch_paths, missing_ranges

@contextmanager
def _lazy_ghi(start: date, end: date, time_res: str, extent, coarsen, batch_days: int, variable: str | None):

    if variable is None:
        variable = VARIABLES[time_res]

    with tempfile.TemporaryDirectory(prefix="ghi_batches_") as tmp:
        batch_dir = Path(tmp)
        batch_paths, missing_ranges = _fetch_in_batches(start, end, time_res, extent, coarsen, variable, batch_dir, batch_days)

        if not batch_paths:
            raise RuntimeError(
                f"No {time_res!r} GHI files found between {start} and {end}"
            )

        with xr.open_mfdataset(batch_paths, combine="by_coords", combine_attrs="override", parallel=False) as combined:
            da = combined[variable].drop_duplicates(dim="time").sortby("time")
            yield da, missing_ranges
        # batch_paths and batch_dir are removed automatically once the
        # TemporaryDirectory context exits below.

def open_ghi(
    start: date,
    end: date,
    time_res: str = "d",
    extent=SYDNEY_EXTENT,
    coarsen=None,
    batch_days=30,
    variable=None,
) -> xr.DataArray:
    """Fetches GHI data for the given range, cropped to `extent`, and returns
    it as an in-memory DataArray (VARIABLES[time_res] by default - pass
    `variable` to fetch a different variable present in the source files,
    e.g. a DNI variable instead of GHI) without writing anything to disk.
    See write_ghi for the file-writing equivalent - both are thin wrappers
    around _lazy_ghi.
    Extent should have 4 values long min, max; lat min, max.
    If coarsen is int n, coarsen data set by averaging over lat, long blocks of size n x n.
    Fetches batch_days worth of files at a time during the fetch itself, but
    since this function's contract is to return the whole range in memory,
    the entire result still gets materialized at once at the end regardless
    of batch_days - see _lazy_ghi. For very large ranges where you don't need
    everything in memory simultaneously, use write_ghi instead, which streams
    straight to disk without that final full materialization.
    Raises if nothing in the range was accessible.

    Example Usage:
    da = open_ghi(start, end, "d")"""
    with _lazy_ghi(start, end, time_res, extent, coarsen, batch_days, variable) as (da, missing_ranges):
        da = da.load()

    if missing_ranges:
        print(f"open_ghi: fetched data, but missing: "
              + ", ".join(f"{s}-{e}" for s, e in missing_ranges))

    return da

def regrid_to_equal_area(
    da: xr.DataArray,
    resolution_km: float = 55,
    method: InterpMethod="linear",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.DataArray:
    """Interpolate ghi data to Albers equal area projection, returning a data array with coordinates
    x y (eastings, northings) as well as latitude, longitude as a meshgrid.
    Default interpolation method is linear, resolution is 55km."""
    # 1. Coordinate transformers
    # WGS84 lat/lon -> GDA2020 / Australian Albers
    to_equal_area = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:9473",
        always_xy=True,
    )

    # Australian Albers -> lat/lon
    to_latlon = Transformer.from_crs(
        "EPSG:9473",
        "EPSG:4326",
        always_xy=True,
    )

    # 2. Determine projected bounds of original domain
    lat = da[lat_name].values
    lon = da[lon_name].values

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x_orig, y_orig = to_equal_area.transform(
        lon_grid,
        lat_grid,
    )

    dx = resolution_km * 1000  # km -> metres

    x_target = np.arange(
        np.nanmin(x_orig),
        np.nanmax(x_orig) + dx,
        dx,
    )

    y_target = np.arange(
        np.nanmin(y_orig),
        np.nanmax(y_orig) + dx,
        dx,
    )

    xx, yy = np.meshgrid(x_target, y_target)

    # 3. Convert equal-area target points back to lat/lon
    target_lon, target_lat = to_latlon.transform(xx, yy)

    # 4. Interpolate original lat/lon data onto those locations
    target_lat_da = xr.DataArray(
        target_lat,
        dims=("y", "x"),
    )

    target_lon_da = xr.DataArray(
        target_lon,
        dims=("y", "x"),
    )

    result = da.interp(
        {
            lat_name: target_lat_da,
            lon_name: target_lon_da,
        },
        method=method,
    )

    # 5. Add projected coordinates
    result = result.assign_coords(
        x=("x", x_target),
        y=("y", y_target),
        latitude=(("y", "x"), target_lat),
        longitude=(("y", "x"), target_lon),
    )

    result.attrs.update(da.attrs)
    result.attrs["regridding"] = (
        f"Interpolated to {resolution_km:g} km "
        "GDA2020 / Australian Albers equal-area grid"
    )

    return result

def download_ghi(
    start: date,
    end: date,
    output_dir: Path,
    time_res: str = "d",
    extent=SYDNEY_EXTENT,
    coarsen=None,
    batch_days=30,
    variable=None,
) -> Path:
    """Fetches GHI data for the given range (see open_ghi) and writes it to a
    new file named ghi_<start>_<end>.nc under `output_dir`.
    Unlike open_ghi, this writes directly from the still-lazy/dask-backed
    combined array (see _lazy_ghi) instead of loading it into memory first.
    Returns the path of the written file.

    Example Usage:
    path = write_ghi(start, end, Path("."), "d")"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ghi_{start:%Y%m%d}_{end:%Y%m%d}.nc"

    with _lazy_ghi(start, end, time_res, extent, coarsen, batch_days, variable) as (da, missing_ranges):
        da.to_netcdf(output_path)

    if missing_ranges:
        print(f"write_ghi: wrote {output_path}, but missing: "
              + ", ".join(f"{s}-{e}" for s, e in missing_ranges))

    return output_path

def _fetch_stations(stations: list[dict], start: date, end: date, time_res: str, batch_days: int):
    """Shared implementation behind open_stations_data/write_stations_data:
    fetches GHI data for multiple stations, opening each source file only
    once for the whole group instead of once per station. thredds.nci.org.au
    doesn't tolerate concurrent connections well (see open_mf notes), so this
    cuts fetch time by avoiding redundant re-downloads rather than by
    parallelising.
    Each entry in `stations` must be a dict with "name", "lat", "long" keys
    (not just a name looked up against the module-level STATIONS list, since
    a station may not be one of those four - callers are responsible for
    supplying the full dict, e.g. via open_elec.build_dict). Fetches
    batch_days worth of files at a time (see open_ghi) so peak memory stays
    bounded to one batch's cropped extent rather than the whole range.
    Returns (stations, dataarrays, missing_ranges): dataarrays is a list
    parallel to `stations`, already loaded into memory as single-point time
    series. Temp batch files are cleaned up before returning either way."""
    variable = VARIABLES[time_res]

    lats = [s["lat"] for s in stations]
    longs = [s["long"] for s in stations]
    extent = (min(longs) - 0.01, max(longs) + 0.01, min(lats) - 0.01, max(lats) + 0.01)

    with tempfile.TemporaryDirectory(prefix="ghi_batches_") as tmp:
        batch_dir = Path(tmp)
        batch_paths, missing_ranges = _fetch_in_batches(start, end, time_res, extent, None, variable, batch_dir, batch_days)

        if not batch_paths:
            raise RuntimeError(
                f"No {time_res!r} GHI files found between {start} and {end}"
            )

        with xr.open_mfdataset(batch_paths, combine="by_coords", combine_attrs="override", parallel=False) as combined:
            combined = combined.drop_duplicates(dim="time").sortby("time")
            dataarrays = [
                combined[variable].sel(latitude=s["lat"], longitude=s["long"], method="nearest").load()
                for s in stations
            ]
        # batch_paths and batch_dir are removed automatically once the
        # TemporaryDirectory context exits below.

    return stations, dataarrays, missing_ranges

def open_stations_data(stations: list[dict], start: date, end: date, time_res: str = "h", batch_days=30) -> list[xr.DataArray]:
    """Fetches GHI data for multiple stations and returns each as an
    in-memory single-point time series DataArray, without writing anything to
    disk. See write_stations_data for the file-writing equivalent - both are
    thin wrappers around the same underlying fetch (_fetch_stations).
    Each entry in `stations` must be a dict with "name", "lat", "long" keys
    (see open_elec.build_dict for how to obtain one for an arbitrary station).
    Returns a list of DataArrays in the same order as `stations`.

    Example Usage:
    das = open_stations_data([station_dict], start, end, "h")"""
    _stations, dataarrays, missing_ranges = _fetch_stations(stations, start, end, time_res, batch_days)

    if missing_ranges:
        print(f"open_stations_data: fetched {len(dataarrays)} station(s), but missing data for: "
              + ", ".join(f"{s}-{e}" for s, e in missing_ranges))

    return dataarrays

def write_stations_data(stations: list[dict], start: date, end: date, output_dir: Path, time_res: str = "h", batch_days=30) -> list[Path]:
    """Fetches GHI data for multiple stations (see open_stations_data) and
    writes one file per station to `output_dir`, named <station>_<start>_<end>.nc,
    each containing only that station's single-point time series.
    Each entry in `stations` must be a dict with "name", "lat", "long" keys.
    Returns the list of written file paths, in the same order as `stations`.

    Example Usage:
    paths = write_stations_data([station_dict], start, end, Path("station_ghi"), "h")"""
    stations, dataarrays, missing_ranges = _fetch_stations(stations, start, end, time_res, batch_days)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for s, da in zip(stations, dataarrays):
        output_path = output_dir / f"{s['name']}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        da.to_netcdf(output_path)
        paths.append(output_path)

    if missing_ranges:
        print(f"write_stations_data: wrote {len(paths)} file(s), but missing data for: "
              + ", ".join(f"{s}-{e}" for s, e in missing_ranges))

    return paths

def checktime(input, start, end, time_res="d", margin=timedelta(days=1)):
    """Check the netCDF file has an entry from all dates between start to end inclusive,
    matching each expected date to the nearest file timestamp within `margin`.
    Returns dict of list of missing dates and dates with invalid data entries."""
    variable = VARIABLES[time_res]
    margin = np.timedelta64(margin)

    with xr.open_dataset(input) as ds:
        file_times = np.atleast_1d(ds["time"].values).astype("datetime64[D]")

        missing = []
        invalid = []
        for when in iter_dates(start, end, time_res):
            np_when = np.datetime64(when, "D")
            diffs = np.abs(file_times - np_when)
            idx = np.argmin(diffs) if diffs.size else None

            if idx is None or diffs[idx] > margin:
                missing.append(when)
                continue

            values = ds[variable].isel(time=idx).values
            if np.isnan(values).all():
                invalid.append(when)

    return {"missing": missing, "invalid": invalid}


if __name__ == "__main__":
    for y in range(2020, 2026):
        start = date(y,1,1)
        end = date(y,12,31)
        data = DataLoader.from_thredds(start, end, time_res="d", extent=AUSTRALIA_EXTENT, coarsen=None,regrid_equal_area=False)
        data.write(f"aus_ghi_new/{y}_raw.nc")
        print(f"finished {y}")
