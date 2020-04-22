"""Extracts source data for one year and station to CSV file for manual inspection."""

import utils
import logging
import constants
from dfply import X, mask

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler()],
)
year = 2019
station = 'SPE00120323'
logging.debug(f"extraction started for year={year} and station={station}")
filename = f"{year}.csv.gz"
by_year_path = '../../data/by_year'
df = utils.df_file(filename, by_year_path)
logging.debug(f"{len(df)} rows extracted for year={year}")
df_station = df >> mask(X.station == station)
logging.debug(f"{len(df_station)} rows for station={station}")
df_prcp = df_station >> mask(X.element == 'PRCP')
logging.debug(f"{len(df_prcp)} rows with precipitation element")
outfile = '../../data/manual_inspection/extract.csv'
df_prcp.to_csv(outfile, index=False)
logging.debug(f"{len(df_prcp)} rows saved to {outfile}")
