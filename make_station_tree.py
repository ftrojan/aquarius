import logging
import utils
import constants

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler(), logging.FileHandler('../../data/logs/make_station_tree.log', 'w')],
)
logging.debug("started")
stations = utils.load_stations()
tree, node_station = utils.make_station_tree(stations)
tree.to_csv('../../data/station/tree.csv', index=False)
node_station.to_csv('../../data/station/node_station.csv', index=False)
logging.debug("completed")
