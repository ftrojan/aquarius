"""Download a specified range of years from ftp.ncdc.noaa.gov/pub/data/ghcn/daily/by_year"""
import utils
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logging.debug("started")
config = utils.get_config()
logging.debug(f"config={config}")
y0 = utils.get_current_year()
y1 = y0
logging.debug(f"the oldest year is {y1}")
data_path = '../../data/by_year'
utils.load_all_years(year_from=y1, year_to=y0, save_dir=data_path)
logging.debug("completed")
