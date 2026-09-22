from datetime import date, datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt
from openelectricity import OEClient, UnitFueltechType, UnitStatusType
from openelectricity.types import DataMetric, DataInterval


from .dataloader import open_stations_data, _split_date_range
ELEC_API_KEY = None
API_KEY = ELEC_API_KEY
AEST = timezone(timedelta(hours=10))
EARLIEST = date(2019, 5, 1)  # no GHI data available before this date
MAX_DAYS_PER_REQUEST = 365  # OpenElectricity's "1d" interval caps requests at 366 days; 365 for margin

STATIONS = [{'code': 'CULCSF',
  'earliest date': datetime(2025, 8, 1, 11, 20, tzinfo=AEST),
  'lat': -35.709592,
  'long': 146.981761,
  'name': 'Culcairn'},
 {'code': 'NEWENSF',
  'earliest date': datetime(2022, 12, 20, 10, 45, tzinfo=AEST),
  'lat': -30.627323,
  'long': 151.556425,
  'name': 'New England'},
 {'code': 'STUBSF',
  'earliest date': datetime(2024, 12, 19, 9, 0, tzinfo=AEST),
  'lat': -32.26254,
  'long': 149.589068,
  'name': 'Stubbo'},
 {'code': 'WELNSF',
  'earliest date': datetime(2024, 5, 21, 15, 25, tzinfo=AEST),
  'lat': -32.514353,
  'long': 148.957634,
  'name': 'Wellington North'}]
# Code of all sola facilities sorted from oldest (most data available) to newest
CODES = ['NYNGAN',
 'BROKENH',
 'MOREESF',
 'ROYALLA',
 'MLSP',
 'BARCSF',
 'GULLRGWF',
 'KSP1',
 'HUGSF',
 'PSF',
 'GANNSF',
 'LRSF',
 'CLARESF',
 'GRIFSF',
 'MANSLR',
 'BNGSF1',
 'SMCSF',
 'HAMISF',
 'WHITSF',
 'BANNSP',
 'DDSF',
 'CSPVPS',
 'COLEASF',
 'RRSF',
 'EMERASF',
 'WRSF1',
 'KARSF',
 'DAYDSF',
 'BNGSF2',
 'WEMENSF1',
 'SRSF',
 'HAYMSF',
 'CHILDSF',
 'TBSF',
 'LILYVASF',
 'OAKEY1SF',
 'BERYLSF',
 'NUMURKSF',
 'HAUGHT1',
 'RUGBYR',
 'CLERMSF',
 'FINLEYSF',
 'OAKEY2SF',
 'NEVERSF',
 'LIMOSF2',
 'YARANSF',
 'BOMENSF',
 'MARYRSF',
 'GOONUMSF',
 'LIMOSF1',
 'DARLSF',
 'KIAMSF1',
 'WARWSF',
 'SUNRSF1',
 'MOLNGSF1',
 'WELLSF1',
 'YATSF1',
 'MIDDLSF1',
 'GLRWNSF',
 'JEMALNG1',
 'CRWASF1',
 'COHUNSF1',
 'MWPS',
 'WINTSF1',
 'ADP',
 'JUNEESF',
 'GNNDHSF',
 'KENNEDY',
 'WAGGNSF',
 'SUNTPSF',
 'GANGARR',
 'HILLSTN',
 'SEBSF',
 'WDGPH',
 'BOLIVAR',
 'MBPS2',
 'METZSF',
 'WOOLGSF',
 'COLUMSF',
 'HVWW',
 'BLUEGSF',
 'WSTWYSF',
 'PAREPW',
 'MOUSF',
 'EDENVSF',
 'MAPS2',
 'AVLSF',
 'NEWENSF',
 'WANDSF',
 'WYASF',
 'TB2SF',
 'GLENSF',
 'MANNSF',
 'WELNSF',
 'GIRGSF',
 'KINGASF',
 'WLWLSF',
 'WUNUSF',
 'MOKOSF',
 'STUBSF',
 'WOLARSF',
 'KERNGSP',
 'ALDGASF',
 'CULCSF',
 'WNSF',
 'MUCRKSF',
 'GOESF1',
 'GUSF',
 'CRWARP',
 'QP',
 'BUSF',
 'BAKING',
 'CESF',
 'CBWWBA',
 'NASF',
 'SKSF',
 'VALDORA']

# True station coordinates
STATIONS = [
    {'code': 'CULCSF', 'lat': -35.709592, 'long': 146.981761, 'name': 'Culcairn'},
    {'code': 'NEWENSF', 'lat': -30.627323, 'long': 151.556425, 'name': 'New England'},
    {'code': 'STUBSF', 'lat': -32.26254, 'long': 149.589068, 'name': 'Stubbo'},
    {'code': 'WELNSF', 'lat': -32.514353, 'long': 148.957634, 'name': 'Wellington North'}]


def build_dict(name: str) -> dict:
    """Takes station name or code and returns the station dict"""
    match = next((s for s in STATIONS if (s["name"] == name)), None)
    if match is not None:
        return {"name": match["name"], "code": match["code"], "lat": match["lat"], "long": match["long"]}

    with OEClient(API_KEY) as client:
        response = client.get_facilities(
            network_id=["NEM"],
            status_id=[UnitStatusType.OPERATING],
            fueltech_id=[UnitFueltechType.SOLAR_UTILITY],
        )

    facility = next((f for f in response.data if (f.name == name or f.code == name)), None)
    if facility is None:
        raise ValueError(f"No operating NEM solar utility facility corresponding to {name!r} found")

    return {
        "name": facility.name,
        "code": facility.code,
        "lat": facility.location.lat,
        "long": facility.location.lng,
    }

def to_naive_datetime(when: date | datetime) -> datetime:
    """Converts a date or datetime into a plain (timezone-naive) datetime.
    A naive input is assumed to already be local time and is returned as-is
    (a bare date is upgraded to midnight).
    Example Usage:
    start = to_naive_datetime(get_earliest_date(station))"""
    if not isinstance(when, datetime):
        when = datetime.combine(when, datetime.min.time())
    if when.tzinfo is not None:
        when = when.astimezone().replace(tzinfo=None)
    return when

def _clamp_to_earliest(when: datetime) -> datetime:
    """Clamps a naive datetime to no earlier than EARLIEST - no GHI data
    exists before that date, so a start date earlier than it (e.g. from
    get_earliest_date(..., all_units=True), which can predate GHI coverage)
    isn't useful for anything paired with GHI anyway.
    Example Usage:
    start = _clamp_to_earliest(to_naive_datetime(get_earliest_date(station)))"""
    return max(when, to_naive_datetime(EARLIEST))

def get_station_data(station: dict, data_metrics: list[DataMetric] = [DataMetric.POWER, DataMetric.ENERGY], get_all: bool = False, start: date | datetime | None = None, end: date | datetime | None = None, interval: DataInterval = "1d") -> pd.DataFrame:
    """Queries a single station's data for the given metrics between start and
    end, and returns it as a DataFrame indexed by time with one column per
    metric (station["code"] is used as the facility_code).
    A facility can have multiple generating units, and/or report each metric
    as its own row rather than one wide row per interval, so the raw response
    can have several rows sharing the same interval (each with only some
    columns populated, others NaN). Grouping by interval and summing collapses
    that down to one row per interval per metric (NaN contributes 0 to sum),
    so callers don't need to handle this themselves.

    The API caps how much date range a single request can cover (366 days
    for interval="1d"), so if [start, end] exceeds MAX_DAYS_PER_REQUEST, this
    fetches it in multiple <=MAX_DAYS_PER_REQUEST-day requests and
    concatenates the results before grouping.

    Example Usage:
    df = get_station_data(target_stations[0], start, end, [DataMetric.POWER, DataMetric.ENERGY])"""
    if get_all:
        start = get_earliest_date(station)
        end = datetime(2025,12,31)

    if start is not None:
        start = _clamp_to_earliest(to_naive_datetime(start))
    if end is not None:
        end = to_naive_datetime(end)

    if start is not None and end is not None and (end - start).days > MAX_DAYS_PER_REQUEST:
        chunks = [
            (to_naive_datetime(chunk_start), to_naive_datetime(chunk_end))
            for chunk_start, chunk_end in _split_date_range(start.date(), end.date(), MAX_DAYS_PER_REQUEST)
        ]
    else:
        chunks = [(start, end)]

    dfs = []
    with OEClient(API_KEY) as client:
        for chunk_start, chunk_end in chunks:
            response = client.get_facility_data(
                network_code="NEM",
                facility_code=station["code"],
                metrics=data_metrics,
                date_start=chunk_start,
                date_end=chunk_end,
                interval = interval
            )
            dfs.append(response.to_pandas())

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    df = df.groupby("interval", as_index=False).sum()
    return df

def get_earliest_date(station: dict, all_units = False) -> datetime | None:
    """Returns earliest date station has data or
    None if none of them have one set. If all_units=True, 
    returns earliest date where all units at this station have data.

    Example Usage:
    earliest = get_earliest_date(STATIONS[0])"""
    with OEClient(API_KEY) as client:
        response = client.get_facilities(network_id=["NEM"], facility_code=station["code"])
        facility = response.data[0]

    first_seen_dates = [u.data_first_seen for u in facility.units if u.data_first_seen is not None]
    if not first_seen_dates:
        return None
    return max(first_seen_dates) if all_units else min(first_seen_dates)

def plot_station_data(df: pd.DataFrame, title: str = None) -> plt.Figure:
    """Plots every metric (column) in df side by side, one subplot per metric,
    each titled with the metric's name.

    Example Usage:
    plot_station_data(df, title="Stubbo")"""
    fig, axes = plt.subplots(nrows=1, ncols=len(df.columns)-1, figsize=(5 * len(df.columns), 4), squeeze=False)
    axes = axes[0]

    for ax, column in zip(axes, df.columns.drop("interval")):
        series = df.dropna(subset=[column])
        ax.plot(series["interval"], series[column], linewidth=0.5)
        ax.set_title(column.capitalize())
        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        ax.tick_params(axis="x", rotation=45)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig

def get_ghi_energy_df(station: dict, start: date | datetime | None = None, end: date | datetime | None = None, ghi_da = None, time_res: str = "d", standardise=False) -> pd.DataFrame:
    """Builds a DataFrame of standardized daily energy (from the
    OpenElectricity API) and standardized GHI (fetched live via
    dataloader.open_stations_data, using this station's own coordinates.
   
    Example Usage:
    df = get_ghi_energy_df(STATIONS[1])
    df = get_ghi_energy_df(STATIONS[1], start=date(2024, 1, 1), end=date(2024, 12, 31))"""
    if start is None:
        start = date(2025, 12, 1)
    if end is None:
        end = date(2025, 12, 31)
    start = _clamp_to_earliest(to_naive_datetime(start))
    end = to_naive_datetime(end)

    energy_df = get_station_data(station, data_metrics=[DataMetric.ENERGY], start=start, end=end)
    energy = energy_df.set_index("interval")["energy"]
    energy.index = pd.to_datetime(energy.index).normalize()
    std_energy = (energy - energy.mean()) / energy.std() if standardise else energy

    station_dict = build_dict(station["name"])
    if ghi_da is None:
        ghi_da = open_stations_data([station_dict], start.date(), end.date(), time_res=time_res)[0]
    ghi = ghi_da.to_pandas()
    ghi.index = pd.to_datetime(ghi.index).normalize()
    ghi = ghi.loc[str(start):str(end)]
    std_ghi = (ghi - ghi.mean()) / ghi.std() if standardise else ghi

    return pd.concat({"ghi": std_ghi, "energy": std_energy}, axis=1, sort=True).dropna()

