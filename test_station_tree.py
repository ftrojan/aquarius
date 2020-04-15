import logging
import pandas as pd
import constants

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler()],
)


def test_station_tree():
    tree = pd.read_csv('../../data/station/tree.csv').set_index('node_name')
    logging.debug(f"tree {tree.shape} with columns {tree.columns}")
    node_station = pd.read_csv('../../data/station/node_station.csv').set_index('node_name')
    logging.debug(f"node_station {node_station.shape} with columns {node_station.columns}")
    value = "Czech Republic (country in Europe)"
    num_stations = tree.at[value, 'num_stations']
    stids = node_station.loc[value, 'station'].values
    assert len(stids) == num_stations
