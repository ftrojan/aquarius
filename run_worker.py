import logging
import utils
import constants

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler()],
)
engine = utils.sql_engine()
stid = "BR001348003"
utils.do_worker_job(engine, stid)
