"""Functions used for testing"""
import xarray as xr
import numpy as np
import ultraplot as uplt

from .dataloader import build_url, iter_dates


def check_file_timestamps(start, end, time_res="d"):
    """Opens each file for the given range and reports the requested date
    alongside the file's own internal `time` coordinate value(s), to check
    whether files carry correct/unique timestamps or a stale/duplicated one.

    Returns a list of (requested, [file_time_values]) tuples.
    """
    report = []
    for when in iter_dates(start, end, time_res):
        url = build_url(when, time_res)
        try:
            ds = xr.open_dataset(url, engine="netcdf4")
        except OSError as exc:
            print(f"Missing: {when} ({exc})")
            continue

        file_times = list(np.atleast_1d(ds["time"].values))
        report.append((when, file_times))
        print(f"Requested: {when}  ->  file time(s): {file_times}")
        ds.close()

    all_file_times = [t for _, times in report for t in times]
    n_unique = len(set(all_file_times))
    print(
        f"\n{len(all_file_times)} file(s) opened, "
        f"{n_unique} unique time value(s)."
    )
    if n_unique < len(all_file_times):
        print("WARNING: duplicate/non-unique time values detected across files.")

    return report

def zero_out_test_square(ds, center_lat=-37, center_lon=144, half_width=1):
    var_name = next(
        v
        for v in ("surface_global_irradiance", "daily_integral_of_surface_global_irradiance")
        if v in ds.data_vars
    )
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    mask = (
        (np.abs(lat2d - center_lat) <= half_width)
        & (np.abs(lon2d - center_lon) <= half_width)
    )

    data = ds[var_name].values
    data[..., mask] = 0
    ds[var_name].values[:] = data
    return ds


def plot_coverage(da):
    # Find latitude/longitude coordinates regardless of naming convention
    lat_name = "latitude"
    lon_name = "longitude"

    lat = da[lat_name].values
    lon = da[lon_name].values

    # Broadcast to a full grid if lat/lon are 1D dimension coordinates
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    else:
        lat2d, lon2d = lat, lon

    values = np.squeeze(da.values)
    is_null = np.isnan(values)

    fig, ax = uplt.subplots(proj="cyl", figsize=(10, 5))
    ax.format(
        coast=True,
        land=False,
        labels=True,
        title="Data point coverage",
    )
    ax.scatter(
        lon2d[is_null].ravel(),
        lat2d[is_null].ravel(),
        s=1,
        color="grey",
        transform="cyl",
    )
    ax.scatter(
        lon2d[~is_null].ravel(),
        lat2d[~is_null].ravel(),
        s=1,
        color="green",
        transform="cyl",
    )
    uplt.show()
