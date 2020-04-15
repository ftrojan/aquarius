import utils
import logging
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.DEBUG,
    format=utils.logfmt,
    handlers=[logging.StreamHandler(), logging.FileHandler('../../data/logs/incremental_update.log')],
)
current_year = utils.get_current_year()
filename = f"{current_year}.csv.gz"
by_year_path = '../../data/by_year'
utils.download_ghcn_file(filename, save_dir=by_year_path)
engine = create_engine('postgres://postgres:@localhost/ghcn')
utils.extract_one_prcp_to_sql(f'{current_year}.csv.gz', by_year_path, engine, 'prcp')
utils.update_cumprcp(engine)
logging.debug("completed")
