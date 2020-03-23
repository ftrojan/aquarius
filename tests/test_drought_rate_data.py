import logging
import utils
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()],
)


def test_drought_rate_data_single():
    logging.debug("started")
    year = 2020
    logging.debug(f"year={year}")
    stid = "FR009021000"
    logging.debug(f"stid={stid}")
    tstarted = datetime.utcnow()
    rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf = utils.drought_rate_data(stid, year)
    tcompleted = datetime.utcnow()
    logging.debug(f"rdf=DataFrame({rdf.shape})")
    logging.debug(f"cprcp=DataFrame({cprcp.shape})")
    logging.debug(f"curr_drought_rate={curr_drought_rate}")
    logging.debug(f"curr_fillrate={curr_fillrate}")
    logging.debug(f"curr_fillrate_cdf={curr_fillrate_cdf}")
    logging.debug(f"completed in {(tcompleted - tstarted)}")


def test_drought_rate_data_multiple():
    logging.debug("started")
    year = 2020
    logging.debug(f"year={year}")
    stdf = utils.load_stations()
    stations = stdf.index.get_level_values('station')
    stations_selected = stations[stations.str.startswith('FR')]
    logging.debug(f"{len(stations_selected)} stations selected")
    tstarted = datetime.utcnow()
    for stid in tqdm(stations_selected, total=len(stations_selected)):
        logging.debug(f"stid={stid}")
        rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf = utils.drought_rate_data(stid, year)
    tcompleted = datetime.utcnow()
    logging.debug(f"rdf=DataFrame({rdf.shape})")
    logging.debug(f"cprcp=DataFrame({cprcp.shape})")
    logging.debug(f"curr_drought_rate={curr_drought_rate}")
    logging.debug(f"curr_fillrate={curr_fillrate}")
    logging.debug(f"curr_fillrate_cdf={curr_fillrate_cdf}")
    logging.debug(f"completed in {(tcompleted - tstarted)}")
