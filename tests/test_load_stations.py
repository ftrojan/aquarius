import logging
import utils
from dfply import X, group_by, summarize, n

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s#%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()],
)


def test_load_countries():
    cdf = utils.load_countries()
    cont = cdf >> group_by(X.continent_code) >> summarize(num_countries=n(X.country_code))
    logging.debug(cont)
    assert {'country_code', 'continent_code', 'country_name'}.issubset(set(cdf.columns))


def test_load_stations():
    sdf = utils.load_stations()
    cz = sdf['country_code'] == 'EZ'
    for s in sdf.loc[cz, :].itertuples():
        logging.debug(s)
    assert {'station', 'country_code'}.issubset(set(sdf.columns))


def test_load_country_continent():
    ccdf = utils.load_country_continent()
    logging.debug(f"{len(ccdf)} countries parsed")
    cont = ccdf >> group_by(X.Continent_Code, X.Continent_Name) >> summarize(num_countries=n(X.Country_Number))
    logging.debug(cont)
    assert cont.shape[0] == 6
