import utils
import logging
import constants

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler(), logging.FileHandler('../../data/logs/incremental_update.log', 'a')],
)
current_year = utils.get_current_year()
filename = f"{current_year}.csv.gz"
by_year_path = '../../data/by_year'
utils.download_ghcn_file(filename, save_dir=by_year_path)
engine = utils.sql_engine()
utils.extract_one_prcp_to_sql(f'{current_year}.csv.gz', by_year_path, engine, 'prcp')
utils.update_cumprcp(engine)
utils.refresh_drought(engine)
logging.debug("completed")
