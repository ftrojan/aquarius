"""Utility functions shared across the Aquarius project."""
import ftplib
import os
import logging
import gzip
import numpy as np
import pandas as pd
import yaml
import json
from datetime import timedelta, date, datetime
from dfply import (
    X,
    group_by,
    summarize,
    mask,
    n,
    transmute,
    select,
    left_join,
    ungroup,
    arrange,
    mutate,
)
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional, NamedTuple
from geopy.distance import geodesic
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, VArea, Line, VBar
from bokeh.tile_providers import get_provider, STAMEN_TERRAIN


class ECDF:
    """Empirical Cumulative Distribution Function with linear interpolation."""

    def __init__(self):
        self.x_values = None
        self.cdf_values = None

    def fit(self, xdata: np.ndarray, weights: Optional[np.ndarray] = None):
        if weights is None:
            ind_valid = ~np.isnan(xdata)
            xv = xdata[ind_valid]
            values, counts = np.unique(xv, return_counts=True)
            sort_index = np.argsort(values)
            self.x_values = values[sort_index]
            self.cdf_values = (np.cumsum(counts[sort_index]) - 0.5)/np.sum(counts)
        else:
            assert len(xdata) == len(weights)
            ind_valid = ~np.isnan(xdata) & ~np.isnan(weights)
            xv = xdata[ind_valid]
            wv = weights[ind_valid]
            sorter = np.argsort(xv)
            values = xv[sorter]
            sample_weight = wv[sorter]
            weighted_quantiles = (np.cumsum(sample_weight) - 0.5 * sample_weight) / np.sum(sample_weight)
            unique_values, unique_index, unique_counts = np.unique(values, return_index=True, return_counts=True)
            self.x_values = unique_values
            self.cdf_values = weighted_quantiles[unique_index + unique_counts - 1]  # last index instead of first index
        return self

    def eval(self, x: np.ndarray):
        cdf = np.interp(x, xp=self.x_values, fp=self.cdf_values, left=0, right=1)
        return cdf

    def quantile(self, q: np.ndarray):
        assert np.all(q >= 0) and np.all(q <= 1), 'quantiles should be in [0, 1]'
        xq = np.interp(q, xp=self.cdf_values, fp=self.x_values, left=self.x_values[0], right=self.x_values[-1])
        return xq


def download_ghcn_file(ftp_filename: str, save_dir: str):
    logging.debug(f"ftp_filename={ftp_filename}")
    logging.debug(f"save_dir={save_dir}")
    ftp = ftplib.FTP(host='ftp.ncdc.noaa.gov', timeout=10.0, user='anonymous', passwd='passwd')
    logging.debug("FTP server connected")
    ftp.cwd("/pub/data/ghcn/daily/by_year/")
    save_path = os.path.join(save_dir, ftp_filename)
    logging.debug(f"downloading {save_path}")
    with open(save_path, 'wb') as file:
        ftp.retrbinary(f"RETR {ftp_filename}", file.write)
    logging.debug(f"downloaded {save_path}")
    return 1


def unzip_file(filename: str, folder: str):
    read_path = os.path.join(folder, filename)
    logging.debug(f"unzipping {read_path}")
    f = gzip.open(read_path, 'rb')
    file_content = f.read()
    f.close()
    logging.debug(f"unzipped {read_path}")
    return file_content


def df_file(filename: str, folder: str) -> pd.DataFrame:
    # based on https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python
    read_path = os.path.join(folder, filename)
    logging.debug(f"unzipping and reading {read_path}")
    with gzip.open(read_path, 'rb') as f:
        df = pd.read_csv(
            f,
            header=None,
            names=['station', 'dateto', 'element', 'value', 'm_flag', 'q_flag', 's_flag', 'obs_time'],
            parse_dates=['dateto'],
        )
    logging.debug(f"read {read_path}")
    return df


def get_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_all_years(year_from: int, year_to: int, save_dir: str):
    for year in range(year_from, year_to + 1):
        filename = f"{year}.csv.gz"
        download_ghcn_file(filename, save_dir)
    logging.debug("completed")


def extract_one_prcp(filename: str, by_year_path: str, prcp_path: str):
    df = df_file(filename, by_year_path)
    df_sel = df >> mask(X.element == 'PRCP') >> transmute(station=X.station, dateto=X.dateto, prcp=X.value)
    logging.debug(f"{df_sel.shape[0]} out of {df.shape[0]} rows selected")
    year_string = filename.split('.')[0]
    df_sel.to_csv(os.path.join(prcp_path, f"{year_string}.csv"), sep=',', index=False)
    logging.debug(f"{filename} processed")


def extract_one_station_prcp(station: str, filename: str, by_year_path: str, prcp_path: str):
    df = df_file(filename, by_year_path)
    df_sel = df >> mask(X.element == 'PRCP') >> mask(X.station == station) >> \
        transmute(station=X.station, dateto=X.dateto, prcp=X.value)
    logging.debug(f"{df_sel.shape[0]} out of {df.shape[0]} rows selected")
    year_string = filename.split('.')[0]
    df_sel.to_csv(os.path.join(prcp_path, f"{year_string}.csv"), sep=',', index=False)
    logging.debug(f"{filename} processed")


def extract_one_station_startswith(station_startswith: str, filename: str, by_year_path: str, prcp_path: str):
    df = df_file(filename, by_year_path)
    df_sel = df >> mask(X.element == 'PRCP') >> mask(X.station.str.startswith(station_startswith)) >> \
        transmute(station=X.station, dateto=X.dateto, prcp=X.value)
    logging.debug(f"{df_sel.shape[0]} out of {df.shape[0]} rows selected")
    year_string = filename.split('.')[0]
    df_sel.to_csv(os.path.join(prcp_path, f"{year_string}.csv"), sep=',', index=False)
    logging.debug(f"{filename} processed")


def extract_all_prcp(by_year_path: str, prcp_path: str):
    if not os.path.isdir(prcp_path):
        os.makedirs(prcp_path)
    for filename in sorted(os.listdir(by_year_path), reverse=True):
        extract_one_prcp(filename, by_year_path, prcp_path)
    return 1


def extract_all_prcp_station(station: str, by_year_path: str, prcp_path: str):
    if not os.path.isdir(prcp_path):
        os.makedirs(prcp_path)
    for filename in sorted(os.listdir(by_year_path), reverse=True):
        extract_one_station_prcp(station, filename, by_year_path, prcp_path)
    return 1


def extract_all_prcp_station_startswith(station_startswith: str, by_year_path: str, prcp_path: str):
    if not os.path.isdir(prcp_path):
        os.makedirs(prcp_path)
    for filename in sorted(os.listdir(by_year_path), reverse=True):
        extract_one_station_startswith(station_startswith, filename, by_year_path, prcp_path)
    return 1


def ded(prcp: pd.DataFrame) -> date:
    logging.debug(f"{prcp.shape[0]} station*days")
    station_ded = prcp >> group_by(X.station) >> summarize(ded=X.dateto.max())
    logging.debug(f"{station_ded.shape[0]} stations")
    data_end_dt = station_ded['ded'].quantile(0.90)
    data_end_date = date(data_end_dt.year, data_end_dt.month, data_end_dt.day)
    logging.debug(f"data_end_date={data_end_date}")
    return data_end_date


def date_limits(prcp: pd.DataFrame) -> tuple:
    logging.debug(f"{prcp.shape[0]} station*days")
    station_ded = prcp >> group_by(X.station) >> summarize(dsd=X.dateto.min(), ded=X.dateto.max())
    logging.debug(f"{station_ded.shape[0]} stations")
    data_start_date = station_ded['dsd'].quantile(0.10)
    data_end_date = station_ded['ded'].quantile(0.90)
    return data_start_date, data_end_date


def df_prcp(year: int, prcp_path=None) -> pd.DataFrame:
    if prcp_path is None:
        prcp_path = '../../data/prcp_ruzyne'
    filename = os.path.join(prcp_path, f'{year}.csv')
    logging.debug(f"reading {filename}")
    prcp = pd.read_csv(filename, parse_dates=['dateto']) >> arrange(X.dateto, X.station)
    return prcp


def active_stations(prcp: pd.DataFrame, date_valid, config) -> pd.DataFrame:
    prcp_valid = prcp >> mask(X.dateto <= date_valid)
    data_end_date = ded(prcp_valid)
    logging.debug(f"data_end_date={data_end_date}")
    logging.debug(f"active_period_length_days={config['active_period_length_days']}")
    active_start_date = data_end_date - timedelta(days=config['active_period_length_days']-1)
    logging.debug(f"active_start_date={active_start_date}")
    prcp_window = prcp_valid >> mask(X.dateto >= active_start_date)
    prcp_active = prcp_window >> group_by(X.station) >> summarize(num_observed_days=n(X.prcp)) >> arrange(X.station)
    prcp_active['is_active'] = prcp_active['num_observed_days'] >= config['active_period_min_days']
    return prcp_active >> ungroup()


def transpose_to_stations(prcp_path: str, stations_path: str):
    # deprecated - too slow
    all_files = sorted(os.listdir(prcp_path), reverse=True)
    num_files = len(all_files)
    logging.debug(f"{num_files} files in {prcp_path}")
    for i_file, filename in enumerate(all_files):
        year = int(filename.split('.')[0])
        df = df_prcp(year)
        stations = df['station'].unique().sort_values()
        num_stations = len(stations)
        logging.debug(f"{num_stations} stations in {filename}")
        for i_station, station in enumerate(stations):
            df_sel = df >> mask(X.station == station) >> select(X.dateto, X.prcp)
            out_filename = os.path.join(stations_path, f"{station}.csv")
            if os.path.isfile(out_filename):
                df_sel.to_csv(out_filename, mode='a', index=False, header=False)
            else:
                df_sel.to_csv(out_filename, mode='w', index=False, header=True)
            logging.debug(f"file={i_file}/{num_files} station={i_station}/{num_stations} processed")
        logging.debug(f"{filename} processed")
    logging.debug("transpose completed")


def make_recent(data_end_date: date, config) -> pd.Series:
    """Make daily calendar with period taking values True=recent and False=preceding."""
    num_days_recent = 365*config['recent_time_window_years']
    num_days_preceding = 365*config['preceding_time_window_max_years']
    num_days = num_days_recent + num_days_preceding
    date_axis = np.flip(pd.date_range(end=data_end_date, periods=num_days, freq='D'))
    calendar_values = np.concatenate([
        np.ones(num_days_recent, dtype=bool),
        np.zeros(num_days_preceding, dtype=bool),
    ])
    calendar = pd.Series(calendar_values, index=date_axis)
    logging.debug((
        f"calendar with {num_days} days from {date_axis[-1]} to {date_axis[0]} "
        f"with recent period of {num_days_recent} from {date_axis[num_days_recent-1]}"
    ))
    return calendar


def update_drought(df_running: pd.DataFrame, df_update: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """Update drought statistics with time series from a new time period."""
    if df_update.shape[0] > 0:
        assert "station" in df_running.columns
        assert "station" in df_update.columns
        assert "dateto" in df_update.columns
        running_columns = [
            'recent_time_window_days',
            'recent_days_observed',
            'recent_fill_rate',
            'recent_precipitation_mm',
            'recent_precipitation_annual_mean',
            'preceding_time_window_days',
            'preceding_days_observed',
            'preceding_fill_rate',
            'preceding_precipitation_mm',
            'preceding_precipitation_annual_mean',
        ]
        for column in running_columns:
            if column not in df_running.columns:
                df_running[column] = 0
        d1, d2 = date_limits(df_update)
        logging.debug(f"date_limits: {d1} and {d2}")
        calendar_recent = pd.DataFrame({'dateto': calendar[calendar].index})
        recent_start_date = calendar_recent.iat[-1, 0]
        recent_end_date = calendar_recent.iat[0, 0]
        calendar_preceding = pd.DataFrame({'dateto': calendar[~calendar].index})
        preceding_start_date = calendar_preceding.iat[-1, 0]
        preceding_end_date = calendar_preceding.iat[0, 0]
        d1_recent = max(d1, recent_start_date)
        d2_recent = min(d2, recent_end_date)
        recent_delta_days = max((d2_recent - d1_recent).days + 1, 0)
        logging.debug(f"recent_delta_days={recent_delta_days}")
        d1_preceding = max(d1, preceding_start_date)
        d2_preceding = min(d2, preceding_end_date)
        preceding_delta_days = max((d2_preceding - d1_preceding).days + 1, 0)
        logging.debug(f"preceding_delta_days={preceding_delta_days}")
        if (recent_delta_days > 0) or (preceding_delta_days > 0):
            logging.debug("proceeding")
            df_base = df_running[['station']].copy()
            df_update_recent = calendar_recent >> \
                left_join(df_update, by='dateto') >> \
                group_by(X.station) >> \
                summarize(
                    recent_days_observed=n(X.prcp),
                    recent_precipitation_mm=X.prcp.sum()/10,
                )
            if df_update_recent.shape[0] == 0:  # df_update does not intersect recent window
                df_update_recent = df_base.copy()
                df_update_recent['recent_days_observed'] = 0
                df_update_recent['recent_precipitation_mm'] = 0.0
            # logging.debug(df_update_recent.head())
            df_update_preceding = calendar_preceding >> \
                left_join(df_update, by='dateto') >> \
                group_by(X.station) >> \
                summarize(
                    preceding_days_observed=n(X.prcp),
                    preceding_precipitation_mm=X.prcp.sum()/10
                )
            if df_update_preceding.shape[0] == 0:  # df_update does not intersect preceding window
                df_update_preceding = df_base.copy()
                df_update_preceding['preceding_days_observed'] = 0
                df_update_preceding['preceding_precipitation_mm'] = 0.0
            # logging.debug(df_update_preceding.head())
            df_delta = df_base.copy() >> \
                left_join(df_update_recent, by='station') >> \
                left_join(df_update_preceding, by='station')
            df_delta.fillna(value=0, inplace=True)
            assert df_delta.shape[0] == df_running.shape[0]
            recent_time_window_days = df_running.recent_time_window_days + recent_delta_days
            preceding_time_window_days = df_running.preceding_time_window_days + preceding_delta_days
            recent_days_observed = df_running.recent_days_observed + df_delta.recent_days_observed
            preceding_days_observed = df_running.preceding_days_observed + df_delta.preceding_days_observed
            recent_fill_rate = recent_days_observed / recent_time_window_days
            preceding_fill_rate = preceding_days_observed / preceding_time_window_days
            recent_precipitation_mm = df_running.ecent_precipitation_mm + df_delta.recent_precipitation_mm
            preceding_precipitation_mm = df_running.preceding_precipitation_mm + df_delta.preceding_precipitation_mm
            recent_precipitation_annual_mean = recent_precipitation_mm / recent_days_observed * 365
            preceding_prcp_annual_mean = preceding_precipitation_mm / preceding_days_observed * 365
            df_running['recent_time_window_days'] = recent_time_window_days
            df_running['recent_days_observed'] = recent_days_observed
            df_running['recent_fill_rate'] = recent_fill_rate
            df_running['recent_precipitation_mm'] = recent_precipitation_mm
            df_running['recent_precipitation_annual_mean'] = recent_precipitation_annual_mean
            df_running['preceding_time_window_days'] = preceding_time_window_days
            df_running['preceding_days_observed'] = preceding_days_observed
            df_running['preceding_fill_rate'] = preceding_fill_rate
            df_running['preceding_precipitation_mm'] = preceding_precipitation_mm
            df_running['preceding_precipitation_annual_mean'] = preceding_prcp_annual_mean
            df_running['dq_flag'] = (recent_fill_rate >= 0.90) & (preceding_fill_rate >= 0.80)
            df_running['drought_index'] = 100*(1 - recent_precipitation_annual_mean / preceding_prcp_annual_mean)
        else:
            logging.debug("skipping")
    else:
        logging.debug("df_running is empty")
    return df_running


def get_current_year() -> int:
    y0 = date.today().year
    return y0


def get_oldest_year() -> int:
    current_year = get_current_year()
    config = get_config()
    oldest_year = current_year - \
        config['drought_window_years'] - \
        config['recent_time_window_years'] - \
        config['preceding_time_window_min_years']
    return oldest_year


def calculate_drought(
        stations: pd.DataFrame,
        data_end_date: date,
        prcp_path: str,
        out_path: str,
        ) -> pd.DataFrame:
    logging.info(f"{stations.shape[0]} active stations with data_end_date={data_end_date}")
    config = get_config()
    calendar = make_recent(data_end_date, config)
    year_to = calendar.index[0].year
    year_from = calendar.index[-1].year
    years = range(year_to, year_from - 1, -1)
    logging.info(f"processing {len(years)} years from {year_to} back to {year_from}")
    for year in years:
        logging.info(f"year={year}")
        prcp_year = df_prcp(year, prcp_path)
        stations = update_drought(stations, prcp_year, calendar)
    logging.info(f"{stations['dq_flag'].sum()} data quality passed")
    stations.to_csv(f'{out_path}/{data_end_date.isoformat()[:10]}.csv', index=False)
    logging.debug(f"\n{stations.head(10)}")
    aquarius = stations >> mask(X.dq_flag) >> \
        summarize(
           min=X.drought_index.min(),
           p25=X.drought_index.quantile(0.25),
           p50=X.drought_index.quantile(0.50),
           p75=X.drought_index.quantile(0.75),
           max=X.drought_index.max(),
        )
    return aquarius


def load_countries() -> pd.DataFrame:
    countries_file = '../../data/station/ghcnd-countries-continent.txt'
    cdf_list = []
    with open(countries_file, 'r') as file:
        for line in file:
            country_code = line[:2]
            continent_code = line[3:5]
            country_name = line[6:].rstrip()
            cdf_row = (country_code, continent_code, country_name)
            cdf_list.append(cdf_row)
        logging.debug(f"{len(cdf_list)} countries parsed")
    cdf = pd.DataFrame(cdf_list, columns=['country_code', 'continent_code', 'country_name'])
    continent = {
        'EU': 'Europe',
        'AS': 'Asia',
        'AF': 'Africa',
        'NA': 'North America',
        'SA': 'South America',
        'OC': 'Oceania',
        'AN': 'Antarctica',
    }
    cdf['continent_name'] = cdf['continent_code'].apply(lambda x: continent[x])
    return cdf


def load_stations() -> pd.DataFrame:
    stations_file = '../../data/station/ghcnd-stations.txt'
    stations_list = []
    with open(stations_file, 'r') as file:
        for line in file:
            country_code = line[:2]
            station = line[:11]
            latitude = float(line[12:20])
            longitude = float(line[21:30])
            elevation = float(line[31:37])
            station_name = line[41:71].rstrip().lower()
            stations_row = (station, country_code, latitude, longitude, elevation, station_name)
            stations_list.append(stations_row)
        logging.debug(f"{len(stations_list)} stations parsed")
    colnames = ['station', 'country_code', 'latitude', 'longitude', 'elevation', 'station_name']
    sdfbase = pd.DataFrame(stations_list, columns=colnames)
    cdf = load_countries()
    sdf = sdfbase.merge(cdf, how='left', on='country_code').set_index('station')
    return sdf


def load_country_continent() -> pd.DataFrame:
    cc_file = '../../data/station/country-and-continent-codes-list-csv_csv.txt'
    ccdf = pd.read_csv(cc_file, sep=",")
    return ccdf


def chunker(seq, size):
    # from http://stackoverflow.com/a/434328
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def insert_with_progress(df, engine, table_name: str, chunksize=None, reset_index=True):
    if reset_index:
        dfi = df.reset_index()
    else:
        dfi = df
    if chunksize is None:
        chunksize = int(len(dfi) / 10)  # 10%
    with tqdm(total=len(dfi)) as pbar:
        for i, cdf in enumerate(chunker(dfi, chunksize)):
            cdf.to_sql(con=engine, name=table_name, if_exists="append", index=False)
            pbar.update(chunksize)


def extract_one_prcp_to_sql(filename: str, by_year_path: str, engine, table_name: str):
    keys = ['station', 'dateto']
    df = df_file(filename, by_year_path)
    logging.debug(f"dateframe {df.shape} loaded")
    df_sel = df >> mask(X.element == 'PRCP') >> transmute(station=X.station, dateto=X.dateto, prcp_mm=X.value / 10)
    logging.debug(f"prcp data {df_sel.shape} extracted")
    dmin, dmax = (df_sel['dateto'].min(), df_sel['dateto'].max())
    df_sorted = df_sel.set_index(keys)
    sql_mirror = (
        "select station, dateto\n"
        f"from {table_name}\n"
        f"where dateto between '{dmin}' and '{dmax}'\n"
        "order by station, dateto"
    )
    df_mirror = pd.DataFrame(engine.execute(sql_mirror).fetchall(), columns=keys).set_index(keys)
    df_mirror['indb'] = True
    logging.debug(f"mirror data {df_mirror.shape} extracted")
    if df_mirror.shape[0] == 0:
        df_joined = df_sorted
        df_joined['indb'] = False
    else:
        df_joined = df_sorted.join(df_mirror, on=keys, how='left', sort=True)
        df_joined['indb'] = df_joined['indb'].fillna(False)
    if (~df_joined['indb']).sum() > 0:
        df_filtered = df_joined >> mask(~X.indb)
        df_increment = df_filtered >> select(X.prcp_mm)
        logging.debug(f"sql insert to {table_name} in progress")
        insert_with_progress(df_increment, engine, table_name, chunksize=100)
        logging.debug(f"insert to {table_name} completed")
    else:
        logging.debug("increment is empty")


def extract_all_prcp_to_sql(by_year_path: str, engine, table_name: str):
    files = sorted(os.listdir(by_year_path))
    nfiles = len(files)
    for i, filename in enumerate(files):
        logging.debug(f"{i + 1}/{nfiles} {filename}")
        extract_one_prcp_to_sql(filename, by_year_path, engine, table_name)
    logging.debug("extract completed")


def find_topk_nearest(k: int, station, index, x1, sortindex1, x2, sortindex2) -> List[dict]:
    nst = len(index)  # total number of stations
    point = (station.latitude, station.longitude)
    i1 = np.where(x1[sortindex1] == point[0])[0][0]
    i2 = np.where(x2[sortindex2] == point[1])[0][0]
    n1 = 100  # intial perimeter, expert guess, works on ruzyne
    n2 = 100  # intial perimeter, expert guess, works on ruzyne
    inperim = np.zeros(nst, dtype=bool)
    ninp = 1
    while ninp < k + 1:
        i1lb = max(i1 - n1, 0)
        i1ub = min(i1 + n1, nst - 1)
        x1lb, x1ub = (x1[sortindex1][i1lb], x1[sortindex1][i1ub])
        i2lb = max(i2 - n2, 0)
        i2ub = min(i2 + n2, nst - 1)
        x2lb, x2ub = (x2[sortindex2][i2lb], x2[sortindex2][i2ub])
        inperim = (x1 >= x1lb) & (x1 <= x1ub) & (x2 >= x2lb) & (x2 <= x2ub)
        ninp = np.sum(inperim)
        n1 *= 2
        n2 *= 2
    distvec = np.array([geodesic(point, station_point).km for station_point in zip(x1[inperim], x2[inperim])])
    indout = np.argsort(distvec)[1:k + 1]
    result = [{'station': stid, 'dist_km': disti} for stid, disti in zip(index[indout], distvec[indout])]
    return result


def find_nearest_stations(stations: pd.DataFrame) -> Dict[str, List]:
    topk = 3
    x1 = stations['latitude'].values
    x2 = stations['longitude'].values
    sortindex1 = np.argsort(x1)
    sortindex2 = np.argsort(x2)
    result = {}
    for station in tqdm(stations.itertuples(), total=len(stations)):
        topn_list = find_topk_nearest(
            k=topk,
            station=station,
            index=stations.index,
            x1=x1,
            sortindex1=sortindex1,
            x2=x2,
            sortindex2=sortindex2)
        result[station.Index] = topn_list
    return result


def get_nearest_stations() -> Dict[str, list]:
    with open('../../data/station/nearest_stations.json', 'r') as file:
        nearest = json.load(file)
    return nearest


def df_station(station: str, engine=None) -> pd.DataFrame:
    if engine is None:
        engine = create_engine('postgresql://postgres:@localhost/ghcn')
    q = engine.execute(f"select * from prcp where station='{station}' order by dateto").fetchall()
    df = pd.DataFrame(q, columns=['station', 'dateto', 'prcp_mm'])
    return df.set_index(['station', 'dateto'])


def make_day_index(year: int) -> pd.DataFrame:
    """
    Make calendar with day index where 0=last day of the previous year, 1=first day of the year.
    It spans the current year and two previous years, so the range is -730 to +365,
    which is 1095 days for one station and year.
    """
    start_date = date(year-2, 1, 1)
    end_date = date(year, 12, 31)
    zero_date = datetime(year-1, 12, 31)
    date_axis = pd.date_range(start=start_date, end=end_date, freq='D')
    day_index = (date_axis - zero_date).days
    calendar = pd.DataFrame({
        'year': year,
        'dateto': [date(d.year, d.month, d.day) for d in date_axis],
        'day_index': day_index,
    }, columns=['year', 'dateto', 'day_index'])
    # logging.debug(f"calendar with {len(date_axis)} days from {date_axis[0]} to {date_axis[-1]}")
    return calendar


def calc_reference_station_year(prcp: pd.DataFrame, year: int) -> pd.DataFrame:
    keys = ['station', 'dateto']
    day_index = make_day_index(year)
    day_index['station'] = prcp.index[0][0]
    day_index = day_index.set_index(keys)
    ref = day_index.join(prcp)
    ref['cum_prcp'] = np.nancumsum(ref['prcp_mm'].astype(float))
    day_observed = ref['prcp_mm'].notnull()
    cum_days_observed = np.cumsum(day_observed)
    cum_days_available = np.arange(1, len(ref)+1)
    ref['cum_fillrate'] = cum_days_observed / cum_days_available
    ref['reference_prcp'] = ref['cum_prcp'] / ref['cum_fillrate']
    # ref.at[ref['cum_fillrate'] < 0.8, 'reference_prcp'] = np.nan
    return ref


def calc_reference_station(prcp: pd.DataFrame) -> pd.DataFrame:
    years = np.arange(1981, 2010+1)
    ref_list = []
    for year in years:
        ref_year = calc_reference_station_year(prcp, year)
        ref_list.append(ref_year)
    ref = pd.concat(ref_list, axis=0)
    return ref


def reference_quantiles(reference: pd.DataFrame) -> pd.DataFrame:
    qq = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    cdf_prcp = ECDF()
    cdf_fill = ECDF()
    qlist = []
    keys = ['station', 'day_index']
    for gkeys, gref in reference.groupby(keys):
        if gref.empty or gref['reference_prcp'].notnull().sum() == 0:
            qprcp = np.full(5, np.nan)
            qfill = np.full(5, np.nan)
        else:
            cdf_prcp.fit(gref['reference_prcp'], weights=gref['cum_fillrate'])
            qprcp = cdf_prcp.quantile(qq)
            cdf_fill.fit(gref['cum_fillrate'])
            qfill = cdf_fill.quantile(qq)
        row = (*gkeys, *qprcp, *qfill)
        qlist.append(row)
    cols = [
        *keys,
        'prcp_min',
        'prcp_p25',
        'prcp_p50',
        'prcp_p75',
        'prcp_max',
        'fill_min',
        'fill_p25',
        'fill_p50',
        'fill_p75',
        'fill_max',
    ]
    qdf = pd.DataFrame(qlist, columns=cols).set_index(keys)
    return qdf


def calc_reference_quantiles(prcp: pd.DataFrame) -> pd.DataFrame:
    """Composition of calc_reference_station and reference_quantiles."""
    # This makes sure that we do not use the reference dataset directly, just the quantiles
    ref = calc_reference_station(prcp)
    q = reference_quantiles(ref)
    return q


def load_reference_quantiles(station: str, engine) -> pd.DataFrame:
    """Load reference quantiles from database."""
    q = engine.execute(f"select * from reference where station='{station}'").fetchall()
    cols = [
        'station',
        'day_index',
        'prcp_min',
        'prcp_p25',
        'prcp_p50',
        'prcp_p75',
        'prcp_max',
        'fill_min',
        'fill_p25',
        'fill_p50',
        'fill_p75',
        'fill_max',
    ]
    df = pd.DataFrame(q, columns=cols).set_index(keys=['station', 'day_index'])
    return df


def calc_cumprcp(prcp: pd.DataFrame, year: int) -> pd.DataFrame:
    data_end_date = prcp.index.get_level_values('dateto')[-1]
    cprcp = calc_reference_station_year(prcp, year)  # reuse the same code as for the reference
    cprcp.columns = ['year', 'day_index', 'prcp_mm', 'cum_prcp', 'cum_fillrate', 'ytd_prcp']
    idx = pd.IndexSlice
    return cprcp.loc[idx[:, :data_end_date], :]


def drought_index(cum_prcp: float, reference_cum_prcp: np.ndarray) -> float:
    """Calculate drought index from the cumulative precipitation and the reference values."""
    cdf = ECDF()
    cdf.fit(reference_cum_prcp)
    curr_cdf = cdf.eval(np.array(cum_prcp))
    curr_drought_index = 2 * (0.5 - curr_cdf)
    return curr_drought_index


def current_drought_rate(refq: pd.DataFrame, curr_cprcp: pd.Series) -> float:
    if refq.empty:
        curr_drought_rate = np.nan
    else:
        curr_station = refq.index[0][0]
        curr_day_index = curr_cprcp['day_index']
        curr_ytd_prcp = curr_cprcp['ytd_prcp']
        refq_columns = ['prcp_min', 'prcp_p25', 'prcp_p50', 'prcp_p75', 'prcp_max']
        refq_prcp = refq.loc[(curr_station, curr_day_index), refq_columns].values
        if len(refq_prcp) > 0:
            curr_drought_rate = drought_index(curr_ytd_prcp, refq_prcp.flatten())
        else:
            curr_drought_rate = np.nan
    return curr_drought_rate


def current_fillrate_cdf(refq: pd.DataFrame, curr_cprcp: pd.Series) -> float:
    curr_station = refq.index[0][0]
    curr_day_index = curr_cprcp['day_index']
    curr_fillrate = curr_cprcp['cum_fillrate']
    refq_columns = ['fill_min', 'fill_p25', 'fill_p50', 'fill_p75', 'fill_max']
    ref_fillrate = refq.loc[(curr_station, curr_day_index), refq_columns].values
    if len(ref_fillrate) > 0:
        cdf = ECDF()
        cdf.fit(ref_fillrate.flatten())
        curr_fillrate_cdf = cdf.eval(curr_fillrate)
    else:
        curr_fillrate_cdf = np.nan
    return curr_fillrate_cdf


def station_label(station: pd.Series) -> str:
    coords = f"{station.latitude:.3f}, {station.longitude:.3f}, {station.elevation:.0f}"
    stlabel = f"{station.continent_name}/{station.country_name}/{station.station_name} ({coords})"
    return stlabel


def nice_ylim(y: float) -> float:
    """Guess the ylim which is proportional to the value."""
    step = 10.0 ** np.round(np.log10(0.1*y))
    ub = step * np.ceil(y / step)
    return ub


def cum_prcp_plot_matplotlib(
        stlabel: str,
        rdf: pd.DataFrame,
        cprcp: pd.DataFrame,
        curr_drought_rate: float
):
    """
    Plot cumulative precipitation with Matplotlib. Deprecated in favor of cum_prcp_plot.
    :param stlabel:
    :param rdf:
    :param cprcp:
    :param curr_drought_rate:
    :return:
    """
    f = plt.figure(figsize=(12, 12))
    if not rdf.empty and rdf['prcp_min'].notnull().sum() > 0:
        prcp_ub = nice_ylim(rdf['prcp_max'].iloc[-1])
        xx = rdf['dateto']
        plt.fill_between(x=xx, y1=0, y2=rdf['prcp_min'], color='red', linewidth=0.0, alpha=0.5)
        plt.fill_between(x=xx, y1=rdf['prcp_min'], y2=rdf['prcp_p25'], color='orange', linewidth=0.0, alpha=0.5)
        plt.fill_between(x=xx, y1=rdf['prcp_p25'], y2=rdf['prcp_p75'], color='green', linewidth=0.0, alpha=0.5)
        plt.fill_between(x=xx, y1=rdf['prcp_p75'], y2=rdf['prcp_max'], color='cyan', linewidth=0.0, alpha=0.5)
        plt.fill_between(x=xx, y1=rdf['prcp_max'], y2=prcp_ub, color='blue', linewidth=0.0, alpha=0.5)
        plt.plot(xx, rdf['prcp_p50'], c='grey')
    if not cprcp.empty:
        plt.plot(cprcp.index.get_level_values('dateto'), cprcp['ytd_prcp'], c='red', linewidth=3)
    ax = plt.gca()
    ax.set_title(f"{stlabel}: current drought rate is {100 * curr_drought_rate:.0f}%")
    ax.set_ylabel('3rd year cumulative precipitation in mm')
    ax.grid(True)
    return f


def cum_prcp_plot(
        stlabel: str,
        rdf: pd.DataFrame,
        cprcp: pd.DataFrame,
        curr_drought_rate: float
):
    """
    Plot cumulative precipitation with Bokeh.
    :param stlabel:
    :param rdf:
    :param cprcp:
    :param curr_drought_rate:
    :return:
    """
    src_ref = ColumnDataSource(rdf)
    src_cur = ColumnDataSource(cprcp.reset_index())
    p = figure(
        plot_width=800,
        plot_height=800,
        title=f"{stlabel}: current drought index is {100 * curr_drought_rate:.0f}%",
        y_axis_label="3rd year cumulative precipitation in mm",
        x_axis_type='datetime',
    )
    if not rdf.empty and rdf['prcp_min'].notnull().sum() > 0:
        prcp_ub = nice_ylim(rdf['prcp_max'].iloc[-1])
        amin = VArea(x="dateto", y1=0, y2="prcp_min", fill_color="red", fill_alpha=0.5)
        ap25 = VArea(x="dateto", y1="prcp_min", y2="prcp_p25", fill_color="orange", fill_alpha=0.5)
        ap50 = VArea(x="dateto", y1="prcp_p25", y2="prcp_p75", fill_color="green", fill_alpha=0.5)
        ap75 = VArea(x="dateto", y1="prcp_p75", y2="prcp_max", fill_color="cyan", fill_alpha=0.5)
        amax = VArea(x="dateto", y1="prcp_max", y2=prcp_ub, fill_color="blue", fill_alpha=0.5)
        lp50 = Line(x="dateto", y="prcp_p50", line_color='grey', line_width=3)
        p.add_glyph(src_ref, amin)
        p.add_glyph(src_ref, ap25)
        p.add_glyph(src_ref, ap50)
        p.add_glyph(src_ref, ap75)
        p.add_glyph(src_ref, amax)
        rref = p.add_glyph(src_ref, lp50)
        ttp_ref = [
            ("Date", "@dateto{%F}"),
            ("Day Index", "@day_index"),
            ("Precipitation min", "@prcp_min{0.}"),
            ("Precipitation p25", "@prcp_p25{0.}"),
            ("Precipitation p50", "@prcp_p50{0.}"),
            ("Precipitation p75", "@prcp_p75{0.}"),
            ("Precipitation max", "@prcp_max{0.}"),
        ]
        hover_ref = HoverTool(renderers=[rref], tooltips=ttp_ref, formatters={"@dateto": "datetime"})
        p.add_tools(hover_ref)
    if not cprcp.empty:
        lcur = Line(x='dateto', y='ytd_prcp', line_color='red', line_width=3)
        rcur = p.add_glyph(src_cur, lcur)
        ttp_cur = [
            ("Date", "@dateto{%F}"),
            ("Day Index", "@day_index"),
            ("Precipitation that day (mm)", "@prcp_mm{0.}"),
            ("Precipitation 3rd year cumulative observed (mm)", "@cum_prcp{0.}"),
            ("Fill rate 3rd year cumulative", "@cum_fillrate{0.000}"),
            ("Precipitation 3rd year cumulative predicted (mm)", "@ytd_prcp{0.}"),
        ]
        hover_cur = HoverTool(renderers=[rcur], tooltips=ttp_cur, formatters={"@dateto": "datetime"})
        p.add_tools(hover_cur)
    return p


def cum_fillrate_plot(
        stlabel: str,
        rdf: pd.DataFrame,
        cprcp: pd.DataFrame,
        curr_fillrate: float,
        curr_fillrate_cdf: float,
):
    f = plt.figure(figsize=(16, 9))
    if not cprcp.empty:
        plt.plot(cprcp.index.get_level_values('dateto'), cprcp['cum_fillrate'], c='red', linewidth=3)
    if not rdf.empty:
        plt.fill_between(rdf['dateto'], y1=rdf['fill_min'], y2=rdf['fill_max'], color='lightgray', alpha=0.5)
        plt.fill_between(rdf['dateto'], y1=rdf['fill_p25'], y2=rdf['fill_p75'], color='darkgray', alpha=0.5)
        plt.plot(rdf['dateto'], rdf['fill_p50'], color='gray')
    ax = plt.gca()
    ax.set_ylim(0, 1)
    title = f"{stlabel}: current fill rate is {curr_fillrate:.2f} which is {100 * curr_fillrate_cdf:.0f} percentile"
    ax.set_title(title)
    ax.set_ylabel('fill rate')
    ax.grid(True)
    return f


def totals_barchart_matplotlib(dfy: pd.DataFrame):
    """Deprecated in favor of totals_barchart."""
    f = plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_ylabel("annual precipitation in mm")
    ax.set_title(f"Yearly precipitation totals")
    if not dfy.empty:
        xx = dfy['year'].values
        yy = dfy['prcp_mm'].values / 10
        dd = dfy['observed_days']
        x1 = np.min(xx)
        x2 = np.max(xx)
        mx = np.mean(xx)
        my = np.mean(yy)
        plt.bar(xx, yy, width=0.8)
        plt.step(xx, dd, c='red')
        plt.plot([x1, x2], [my, my], color='blue')
        ax.annotate(
            f"{my:.0f}",
            xy=(mx, my),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize='x-large',
            color='blue',
        )
    return f


def totals_barchart(df: pd.DataFrame):
    dfy = df.copy()
    dfy['prcp_mm'] = df.prcp_mm / 10
    dfy['available_days'] = 365 + (dfy.year % 4 == 0)
    dfy['fill_rate'] = dfy.observed_days / dfy.available_days
    dfy['prcp_pred'] = dfy.prcp_mm / dfy.fill_rate
    prcp_pred_mean = dfy.prcp_pred.mean()
    dfy['prcp_pred_mean'] = prcp_pred_mean
    f = figure(
        plot_width=800,
        plot_height=800,
        title=f"Yearly precipitation totals, mean={prcp_pred_mean:.0f}mm",
        y_axis_label="annual precipitation in mm",
    )
    if not dfy.empty:
        src = ColumnDataSource(dfy)
        bobs = VBar(x='year', top='prcp_mm', fill_color='blue', line_color='blue', width=0.8)
        bpre = VBar(x='year', bottom='prcp_mm', top='prcp_pred', fill_color='lightblue', line_color='blue', width=0.8)
        lpre = Line(x='year', y='prcp_pred_mean', line_color='darkblue', line_width=3)
        f.add_glyph(src, bobs)
        f.add_glyph(src, bpre)
        f.add_glyph(src, lpre)
        ttp = [
            ("Year", "@year"),
            ("Precipitation observed (mm)", "@prcp_mm{0.}"),
            ("Observed days", "@observed_days"),
            ("Available days", "@available_days"),
            ("Fill rate", "@fill_rate{0.000}"),
            ("Precipitation predicted (mm)", "@prcp_pred{0.}"),
        ]
        hover_tool = HoverTool(tooltips=ttp)
        f.add_tools(hover_tool)
    return f


def drought_rate_data(stid: str, year: int, engine=None) -> tuple:
    prcp = df_station(stid, engine)
    if not prcp.empty:
        if engine is None:
            refq = calc_reference_quantiles(prcp)
        else:
            refq = load_reference_quantiles(stid, engine)
        data_end_date = prcp.index.get_level_values('dateto')[-1]
        day_index = make_day_index(year)
        rdf = day_index.merge(refq, on='day_index', how='left') >> mask(X.dateto <= data_end_date)
        if engine is None:
            cprcp = calc_cumprcp(prcp, year)
        else:
            cprcp = calc_cumprcp(prcp, year)  # TODO load from db
        if not cprcp.empty:
            curr_cprcp = cprcp.iloc[-1, :]
            curr_fillrate = curr_cprcp['cum_fillrate']
            if refq.empty:
                curr_drought_rate = np.nan
                curr_fillrate_cdf = np.nan
            else:
                curr_drought_rate = current_drought_rate(refq, curr_cprcp)
                curr_fillrate_cdf = current_fillrate_cdf(refq, curr_cprcp)
        else:
            curr_drought_rate = np.nan
            curr_fillrate = np.nan
            curr_fillrate_cdf = np.nan
    else:
        rdf = pd.DataFrame()
        cprcp = pd.DataFrame()
        curr_drought_rate = np.nan
        curr_fillrate = np.nan
        curr_fillrate_cdf = np.nan
    return rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf


def sql_engine():
    engine = create_engine('postgres://postgres:@localhost/ghcn')
    return engine


def get_stations_noref(engine, stations: pd.DataFrame) -> pd.DataFrame:
    cols = ['station', 'dispatched_at', 'completed_at']
    sql_noref = (
        f"select {', '.join(cols)}\n"
        "from reference_job\n"
        "where completed_at is null"
    )
    station_noref = pd.DataFrame(engine.execute(sql_noref).fetchall(), columns=cols).set_index('station')
    station_coord = station_noref.join(stations)
    lat = station_coord.latitude
    lon = station_coord.longitude
    center_point = (49.9629345, 14.0600897)  # x=14.0600897&y=49.9629345 = Beroun Vinicna 1005
    station_coord['perimeter_km'] = np.array([geodesic(center_point, station_point).km for station_point in zip(lat, lon)])
    return station_coord.sort_values(by='perimeter_km')


def ded_prcp(engine) -> date:
    sql_query = (
        "select max(dateto) as ded\n"
        "from prcp"
    )
    df = pd.DataFrame(engine.execute(sql_query).fetchall(), columns=['ded'])
    data_end_dt = df['ded'].iat[0]
    data_end_date = date(data_end_dt.year, data_end_dt.month, data_end_dt.day)
    logging.debug(f"data_end_date={data_end_date}")
    return data_end_date


def ded_cump(engine, year: int) -> tuple:
    sql_query = (
        "select max(dateto) as ded, max(day_index) as dei\n"
        "from cumprcp\n"
        f"where year={year}"
    )
    df = pd.DataFrame(engine.execute(sql_query).fetchall(), columns=['ded', 'dei'])
    data_end_dt = df['ded'].iat[0]
    data_end_index = df['dei'].iat[0]
    if data_end_dt is None:
        data_end_date = None
    else:
        data_end_date = date(data_end_dt.year, data_end_dt.month, data_end_dt.day)
    logging.debug(f"cump_end_date={data_end_date}")
    return data_end_date, data_end_index


def prcp_dateto(engine, dateto: date) -> pd.DataFrame:
    """Select all rows from prcp table as of dateto."""
    sql_query = f"select station, cast(prcp_mm as float) as prcp_mm from prcp where dateto=date'{dateto.isoformat()}'"
    logging.debug(sql_query)
    df = pd.DataFrame(engine.execute(sql_query).fetchall(), columns=['station', 'prcp_mm'])
    return df


def increment_cumprcp(engine, year: int, day_index: int, dateto: date, cum_days_available: int):
    """Insert new records to cumprcp for the spacified day assuming that the previous day is there."""
    cols_previous = ['station', 'year', 'day_index', 'dateto', 'cum_days_observed', 'cum_prcp']
    sql_previous = f"select {', '.join(cols_previous)} from cumprcp where year={year} and day_index={day_index - 1}"
    cumprcp_previous = pd.DataFrame(engine.execute(sql_previous).fetchall(), columns=cols_previous)
    assert not cumprcp_previous.empty
    prcp = prcp_dateto(engine, dateto)
    cols_both = ['station', 'year']
    cumprcp = cumprcp_previous[cols_both].merge(prcp, how='left', on='station')
    cumprcp['day_index'] = day_index
    cumprcp['dateto'] = dateto
    cumprcp['flag_observed'] = cumprcp.prcp_mm.notnull()
    cumprcp['cum_days_observed'] = cumprcp_previous.cum_days_observed + cumprcp.flag_observed
    cumprcp['cum_fillrate'] = cumprcp.cum_days_observed / cum_days_available
    cumprcp['cum_prcp'] = cumprcp_previous.cum_prcp + cumprcp.prcp_mm.fillna(0)
    cumprcp['cum_prcp_pred'] = cumprcp.cum_prcp / cumprcp.cum_fillrate
    cols_out = [
        'station',
        'year',
        'day_index',
        'dateto',
        'flag_observed',
        'cum_days_observed',
        'cum_fillrate',
        'cum_prcp',
        'cum_prcp_pred',
    ]
    insert_with_progress(cumprcp[cols_out], engine, table_name='cumprcp', reset_index=False)


def update_cumprcp(engine):
    """Update table cumprcp if new data is available in prcp table."""
    stations = load_stations()
    prcp_end_date = ded_prcp(engine)
    year = prcp_end_date.year
    day_index = make_day_index(year)
    first_day_index = day_index['day_index'].iat[0]
    cump_end_date, cump_end_index = ded_cump(engine, year)
    if cump_end_date is None:  # create new year skeleton
        dateto0 = day_index['dateto'].iat[0]
        logging.debug(dateto0)
        prcp0 = prcp_dateto(engine, dateto0)
        cump0 = stations >> left_join(prcp0, by='station')
        flag_observed = cump0.prcp_mm.notnull()
        skeleton = pd.DataFrame({
            'station': stations.index,
            'year': year,
            'day_index': first_day_index,
            'dateto': dateto0,
            'flag_observed': flag_observed,
            'cum_days_observed': flag_observed.astype(int),
            'cum_fillrate': flag_observed.astype(float),
            'cum_prcp': cump0.prcp_mm.fillna(0),
            'cum_prcp_pred': cump0.prcp_mm,
        })
        insert_with_progress(skeleton, engine, table_name='cumprcp', reset_index=False)
        cump_end_date = dateto0
    day_index_todo = day_index.loc[(day_index.dateto > cump_end_date) & (day_index.dateto <= prcp_end_date), :]
    for x in day_index_todo.itertuples():
        logging.debug(x)
        cum_days_available = x.day_index - first_day_index + 1
        increment_cumprcp(engine, x.year, x.day_index, x.dateto, cum_days_available)
    logging.debug("completed")


def do_worker_job(engine, station_id: str):
    if station_id:
        prcp = df_station(station_id)
        if not prcp.empty:
            refq = calc_reference_quantiles(prcp)
            if not refq.empty:
                insert_with_progress(refq, engine, table_name='reference', chunksize=2000)
    return


def make_station_tree(stations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """All nodes of the stations tree are searcheable in the autocomplete by node_name."""

    def node_name_station(station_record: NamedTuple) -> str:
        name = f"{station_record.station_name} (station in {station_record.country_name})"
        return name

    def node_name_country(station_record: NamedTuple) -> str:
        name = f"{station_record.country_name} (country in {station_record.continent_name})"
        return name

    station_name = pd.Series([node_name_station(x) for x in stations.itertuples()])
    if station_name.is_unique:
        logging.debug("tree nodes are unique")
    else:
        logging.error("tree nodes are not unique")
        freq = station_name.value_counts()
        ndup = np.sum(freq > 1)
        logging.debug(f"{ndup} duplicated names")
        for index, value in tqdm(freq[:ndup].iteritems(), total=ndup):
            # logging.debug(f"{index}: {value}x")
            # deduplication - add i/n at the end of each name
            dupidx = np.flatnonzero(station_name == index)
            for i, (idx, ndname) in enumerate(station_name.iloc[dupidx].iteritems()):
                # logging.debug(f"{idx}: {ndname} {i+1}/{value}")
                station_name.at[idx] = f"{ndname} {i+1}/{value}"
    country_name = pd.Series([node_name_country(x) for x in stations.itertuples()])
    continent_name = stations['continent_name']
    node_name = pd.concat([station_name, country_name, continent_name], axis=0)
    stidx = stations.index.values
    station = np.concatenate((stidx, stidx, stidx), axis=0)
    node_station = pd.DataFrame({
        'node_name': node_name,
        'station': station,
    })
    nvc = node_station['node_name'].value_counts()
    tree = pd.DataFrame({
        'node_name': nvc.index,
        'num_stations': nvc.values,
    })
    return tree, node_station


def refresh_drought(engine):
    """Refresh table drought from cumprcp and reference for the last day_index available."""
    sql_year = "select max(year) from cumprcp"
    max_year = engine.execute(sql_year).fetchone()[0]
    logging.debug(f"max_year={max_year}")
    sql_index = f"select max(day_index) from cumprcp where year = {max_year}"
    max_day_index = engine.execute(sql_index).fetchone()[0]
    logging.debug(f"max_day_index={max_day_index}")
    last_cumprcp_cols = ["station", "cum_prcp_pred", "cum_fillrate"]
    sql_last_cumprcp = (
        f"select {', '.join(last_cumprcp_cols)} "
        f"from cumprcp where year = {max_year} and day_index = {max_day_index} "
        f"and {' and '.join([c + ' is not null' for c in last_cumprcp_cols])}"
    )
    logging.debug(sql_last_cumprcp)
    last_cumprcp = pd.DataFrame(
        engine.execute(sql_last_cumprcp).fetchall(), columns=last_cumprcp_cols
    ).set_index('station')
    logging.debug(f"last_cumprcp={last_cumprcp.shape}")
    reference_prcp_cols = ["prcp_min", "prcp_p25", "prcp_p50", "prcp_p75", "prcp_max"]
    last_reference_cols = ["station"] + reference_prcp_cols
    sql_last_reference = (
        f"select {', '.join(last_reference_cols)} "
        f"from reference where day_index = {max_day_index} "
        f"and {' and '.join([c + ' is not null' for c in reference_prcp_cols])}"
    )
    logging.debug(sql_last_reference)
    last_reference = pd.DataFrame(
        engine.execute(sql_last_reference).fetchall(), columns=last_reference_cols
    ).set_index('station')
    logging.debug(f"last_reference={last_reference.shape}")
    drought = last_reference.join(last_cumprcp, how='inner')
    logging.debug(f"drought={drought.shape}")
    drought['drought_index'] = [drought_index(x.cum_prcp_pred, x[reference_prcp_cols]) for s, x in drought.iterrows()]
    engine.execute("truncate table drought")
    out_df = drought.reset_index() >> select(X.station, X.drought_index, X.cum_prcp_pred, X.cum_fillrate)
    logging.debug(f"out_df={out_df.shape}")
    insert_with_progress(out_df, engine, table_name='drought', reset_index=False)


def get_drought(engine) -> pd.DataFrame:
    cols = ["station", "drought_index", "cum_prcp_pred", "cum_fillrate"]
    df = pd.DataFrame(engine.execute("select * from drought").fetchall(), columns=cols).set_index("station")
    return df


def get_drought_stats(drought: pd.DataFrame) -> tuple:
    """Calculate stats from the not aggregated drought data."""
    flag_dry = (drought.drought_index >= 0.5)
    num_dry = np.sum(flag_dry)
    num_stations = len(drought)
    if num_stations > 0:
        drought_rate = num_dry / num_stations
    else:
        drought_rate = np.nan
    return num_stations, num_dry, drought_rate


def aggregate_drought(drought: pd.DataFrame, agg_stations=50) -> pd.DataFrame:
    """Return aggd, num_dry, num_stations."""
    drought = drought.sort_values(by='drought_index')
    num_stations = len(drought)
    bucket_size = np.ceil(num_stations / agg_stations)
    drought['bucket_mark'] = (np.arange(0, num_stations) // bucket_size + 1) * bucket_size
    aggd = drought >> group_by(X.bucket_mark) >> summarize(
        drought_index=X.drought_index.mean(),
        cum_fillrate=X.cum_fillrate.mean(),
        bucket_size=n(X.drought_index),
    )
    aggd = aggd.set_index('bucket_mark')
    aggd.index = aggd.index.set_names('index')
    return aggd


def drought_add_facecolor(drought: pd.DataFrame) -> pd.DataFrame:
    """Add facecolor column before drought plot."""
    bar_colors = np.array(['blue', 'cyan', 'green', 'orange', 'red'])
    drought_index_band = pd.cut(drought.drought_index, [-1.0, -0.99, -0.5, +0.5, +0.99, +1.001], right=False)
    drought['facecolor'] = bar_colors[drought_index_band.cat.codes.values]
    if "bucket_size" in drought.columns:
        drought['barwidth'] = drought['bucket_size'] * 0.8
    else:
        drought['barwidth'] = 0.8
    return drought


def drought_rate_plot_core(drought: pd.DataFrame, drought_stats: tuple, tooltips: list):
    num_stations, num_dry, drought_rate = drought_stats
    src = ColumnDataSource(drought)
    p = figure(
        plot_width=1600,
        plot_height=600,
        title=f"Drought rate is {100 * drought_rate:.0f}%={num_dry}/{num_stations}.",
        y_axis_label="Drought Index",
        x_axis_label="Station Rank by Drought Index",
    )
    p.vbar(
        x="index",
        top="drought_index",
        width="barwidth",
        alpha="cum_fillrate",
        fill_color="facecolor",
        line_color=None,
        source=src)
    p.y_range.start = -1.0
    p.y_range.end = +1.0
    p.add_tools(HoverTool(tooltips=tooltips))
    p.xgrid.grid_line_color = None
    return p


def drought_rate_plot(drought: pd.DataFrame):
    drought_stats = get_drought_stats(drought)
    ttp = [
        ("cum_prcp_predicted", "@cum_prcp_pred{00.}"),
        ("cum_fillrate", "@cum_fillrate{0.00}"),
        ("id", "@station"),
        ("station", "@station_name"),
        ("country", "@country_name"),
        ("elevation", "@elevation{0.}"),
    ]
    plotdf = drought.sort_values(by='drought_index').reset_index()
    p = drought_rate_plot_core(plotdf, drought_stats, ttp)
    return p


def drought_rate_plot_agg(drought: pd.DataFrame, agg_stations=50):
    drought_stats = get_drought_stats(drought)
    aggd = aggregate_drought(drought, agg_stations=agg_stations)
    ttp = [
        ("Mean drought index", "@drought_index{0.000}"),
        ("Bucket size", "@bucket_size"),
        ("Mean fill rate", "@cum_fillrate{0.000}"),
    ]
    aggd = drought_add_facecolor(aggd)
    p = drought_rate_plot_core(aggd, drought_stats, ttp)
    return p


def update_yearly_totals(year: int):
    prcp_path = '../../data/prcp'
    out_file = '../../data/yearly_totals/prcp_totals.csv'
    logging.debug(f"current_year={year}")
    prcp = df_prcp(year, prcp_path)
    totals = prcp >> group_by(X.station) >> summarize(
        prcp_mm=X.prcp.sum(),
        observed_days=n(X.prcp),
    ) >> mutate(year=year)
    logging.debug(f"{len(totals)} totals calculated")
    df1 = pd.read_csv(out_file)
    logging.debug(f"{len(df1)} records read from {out_file}")
    df2 = df1 >> mask(X.year < year)
    logging.debug(f"{len(df2)} records after deleting year {year}")
    logging.debug(f"{len(totals)} records of current year {year} to be added")
    df_totals = pd.concat([df2, totals], axis=0).sort_values(by=['station', 'year'])
    df_totals.to_csv(out_file, index=False)
    logging.debug(f"{len(df_totals)} records saved to {out_file}")


def wgs84_epsg3857(wgs84: tuple) -> tuple:
    """
    Convert (longitude, latitude) coordinates in WGS84 to EPSG 3857 Mercator coordinates.

    See https://epsg.io/3857 for the definition.
    See https://epsg.io/transform#s_srs=4326&t_srs=3857&x=-4.2325468&y=50.4211273 to check.
    """
    lon, lat = wgs84
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x / lon
    y = 180.0 / np.pi * np.log(np.tan(np.pi / 4.0 + lat * (np.pi / 180.0) / 2.0)) * scale
    return x, y


def good_map_limits(x: pd.Series, margin_rel=0.1, margin_abs=50e3) -> tuple:
    """Calculate good map limits for the given dimesion."""
    xmin = x.min()
    xmax = x.max()
    xdel = xmax - xmin
    xoff = max(margin_rel * xdel, margin_abs)
    return xmin - xoff, xmax + xoff


def drought_map(dfm: pd.DataFrame):
    """Draw Bokeh map of points"""
    dfm['alpha'] = 0.3 + 0.7 * dfm.cum_fillrate.fillna(0)
    xy = [wgs84_epsg3857((s.longitude, s.latitude)) for s in dfm.itertuples()]
    dfm['x'] = [t[0] for t in xy]
    dfm['y'] = [t[1] for t in xy]
    tile_provider = get_provider(STAMEN_TERRAIN)
    xrange = good_map_limits(dfm.x)
    yrange = good_map_limits(dfm.y)
    col_source = ColumnDataSource(dfm)
    p = figure(
        plot_width=1600,
        plot_height=800,
        x_range=xrange,
        y_range=yrange,
        x_axis_type="mercator",
        y_axis_type="mercator",
    )
    p.add_tile(tile_provider)
    p.circle(x='x', y='y', size=15, color='facecolor', alpha='alpha', source=col_source)
    ttp = [
        ("cum_prcp_predicted", "@cum_prcp_pred{00.}"),
        ("cum_fillrate", "@cum_fillrate{0.00}"),
        ("id", "@station"),
        ("station", "@station_name"),
        ("country", "@country_name"),
        ("elevation", "@elevation{0.}"),
    ]
    hover_tool = HoverTool(tooltips=ttp)
    p.add_tools(hover_tool)
    return p
