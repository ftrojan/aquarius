"""Update cumprcp table with increment from prcp table."""
import logging
import utils

logging.basicConfig(
    level=logging.DEBUG,
    format=utils.logfmt,
    handlers=[logging.StreamHandler()],
)
logging.debug("started")
engine = utils.sql_engine()
utils.update_cumprcp(engine)
logging.debug("completed")
