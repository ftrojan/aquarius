"""Creates clean initial table reference_job."""

import pandas as pd
import logging
import utils

logging.basicConfig(
    level=logging.DEBUG,
    format=utils.logfmt,
    handlers=[logging.StreamHandler()],
)
logging.debug("started")
engine = utils.sql_engine()
response = engine.execute("truncate table reference_job")
logging.debug(response)
response = engine.execute("truncate table reference")
logging.debug(response)
stations = utils.load_stations()
df_init = pd.DataFrame({
    'station': stations.index.get_level_values('station'),
    'dispatched_at': pd.NaT,
    'completed_at': pd.NaT,
}).set_index('station')
utils.insert_with_progress(df_init, engine, table_name='reference_job', chunksize=100)
logging.debug("completed")
