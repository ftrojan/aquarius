import pytest
import utils


engine = utils.sql_engine()
stations = utils.load_stations()
testdata = [
    ("canberra forestry", 2020),
    ("ruzyne", 2020),
]


@pytest.mark.parametrize("station_search, year", testdata)
def test_drought_rate_data(station_search, year):
    stids = utils.get_station_ids_by_name(station_search, stations)
    stid = stids[0]
    stlabel = utils.station_label(stations.loc[stid, :])
    rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf = utils.drought_rate_data(stid, year, engine)
    f_prcp = utils.cum_prcp_plot(stlabel, rdf, cprcp, curr_drought_rate)
    assert not rdf.empty
    assert not cprcp.empty
    assert f_prcp is not None
