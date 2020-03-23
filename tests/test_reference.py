import logging
import utils

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()],
)


def test_calc_reference_station_year():
    year = 2020
    stid = 'EZM00011518'
    prcp = utils.df_station(stid)
    ref = utils.calc_reference_station_year(prcp, year)
    assert ref.shape[0] == 1096


def test_calc_reference_station():
    stid = 'EZM00011518'
    prcp = utils.df_station(stid)
    ref = utils.calc_reference_station(prcp)
    assert ref['year'].min() == 1981
    assert ref['year'].max() == 2010
