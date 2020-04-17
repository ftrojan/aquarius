"""Update cumprcp table with increment from prcp table."""
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
utils.update_cumprcp(engine)
logging.debug("completed")
