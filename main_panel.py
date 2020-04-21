"""
Main dashboard endpoint.

Use via `panel serve main_panel.py`
"""
import panel as pn
import logging
import pandas as pd
import datetime
import utils
import constants

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler()],
)
stations = utils.load_stations()
current_year = datetime.date.today().year
totals = pd.read_csv('../../data/yearly_totals/prcp_totals.csv')
tree = pd.read_csv('../../data/station/tree.csv').set_index('node_name')
node_station = pd.read_csv('../../data/station/node_station.csv').set_index('node_name')
options = list(tree.index)
logging.debug(options)
autocomplete = pn.widgets.AutocompleteInput(
    name='Autocomplete Input', options=options,
    placeholder='Station, Country or Continent')
input_year = pn.widgets.TextInput(name='Year', value=str(current_year))
engine = utils.sql_engine()
drought = utils.get_drought(engine).join(stations)


@pn.depends(autocomplete.param.value, input_year.param.value)
def p_drought_plot(autocomplete_value: str, year_value: str):
    year = int(year_value)
    if autocomplete_value and year:
        num_stations = tree.at[autocomplete_value, 'num_stations']
        if num_stations == 1:
            stid = node_station.loc[autocomplete_value, 'station']
            stlabel = utils.station_label(stations.loc[stid, :])
            logging.debug(f"calling drought_rate_data with stid={stid} and year={year}")
            rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf = \
                utils.drought_rate_data(stid, year, engine=engine)
            f_prcp = utils.cum_prcp_plot(stlabel, rdf, cprcp, curr_drought_rate)
            dft = totals.loc[totals['station'] == stid, :]
            f_totals = utils.totals_barchart(dft)
            # utils.cum_fillrate_plot(stlabel, rdf, cprcp, curr_fillrate, curr_fillrate_cdf)
            row = pn.Row(f_prcp, f_totals)
        else:  # num_stations > 1
            stids = node_station.loc[autocomplete_value, ['station']]
            df = stids.set_index('station').join(drought).reset_index()
            max_stations = 200
            if num_stations <= max_stations:
                p = utils.drought_rate_plot(df)
            else:
                p = utils.drought_rate_plot_agg(df)
            row = pn.pane.Bokeh(p)
    else:  # empty autocomplete or not valid year - assume world at current_year
        p = utils.drought_rate_plot_agg(drought)
        row = pn.pane.Bokeh(p)
    return row


controls = pn.Row(autocomplete, input_year)
app = pn.Column(controls, p_drought_plot)
app.show()
app.servable()
