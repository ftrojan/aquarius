"""Refresh table drought from cumprcp and reference for the last day_index available."""
import logging
import utils
import constants

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('../../data/logs/update_cumprcp.log', 'a'),
    ],
)
logging.debug("started")
engine = utils.sql_engine()
utils.refresh_drought(engine)
logging.debug("completed")
