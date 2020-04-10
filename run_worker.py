import logging
import utils

logging.basicConfig(
    level=logging.DEBUG,
    format=utils.logfmt,
    handlers=[logging.StreamHandler()],
)
engine = utils.sql_engine()
stid = "BR001348003"
utils.do_worker_job(engine, stid)
