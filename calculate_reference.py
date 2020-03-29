"""Fills reference table with data. Parallel job using dask."""

import pandas as pd
# import dask.distributed
import logging
from typing import NamedTuple
import utils
from tqdm import tqdm


def reference_one_station(station_tuple: NamedTuple) -> pd.DataFrame:
    # logging.debug(station_tuple)
    prcp = utils.df_station(station_tuple.Index)
    if prcp.empty:
        referenceq = None
    else:
        referenceq = utils.calc_reference_quantiles(prcp)
    return referenceq


logging.basicConfig(
    level=logging.DEBUG,
    format=utils.logfmt,
    handlers=[logging.StreamHandler()],
)
logging.debug("started")
engine = utils.sql_engine()
# client = dask.distributed.Client()
stations = utils.load_stations()
refq_list = []
limit = 20
for station in tqdm(stations.head(limit).itertuples(), total=limit):
    refq = reference_one_station(station)
    refq_list.append(refq)
reference = pd.concat(refq_list, axis=0)
logging.debug(reference.dtypes)
logging.debug(reference.shape)
logging.debug("completed")
