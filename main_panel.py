"""
Main dashboard endpoint.

Use via `panel serve main_panel.py`
"""
import panel as pn
import utils
import logging
import pandas as pd
import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format=utils.logfmt,
    handlers=[logging.StreamHandler()],
)
stations = utils.load_stations()
current_year = datetime.date.today().year
totals = pd.read_csv('../../data/yearly_totals/prcp_totals.csv')
options = list(stations['station_name'])
logging.debug(options)
autocomplete = pn.widgets.AutocompleteInput(
    name='Autocomplete Input', options=options,
    placeholder='Station, Country or Continent')
input_year = pn.widgets.TextInput(name='Year', value=str(current_year))
engine = utils.sql_engine()


@pn.depends(autocomplete.param.value, input_year.param.value)
def p_drought_plot(autocomplete_value: str, year_value: str):
    year = int(year_value)
    if autocomplete_value and year:
        stids = utils.get_station_ids_by_name(autocomplete_value, stations)
        if len(stids) > 0:
            stid = stids[0]
            stlabel = utils.station_label(stations.loc[stid, :])
            logging.debug(f"calling drought_rate_data with stid={stid} and year={year}")
            rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf = \
                utils.drought_rate_data(stid, year, engine=engine)
            f_prcp = utils.cum_prcp_plot(stlabel, rdf, cprcp, curr_drought_rate)
            dft = totals.loc[totals['station'] == stid, :]
            f_totals = utils.totals_barchart(dft)
            # utils.cum_fillrate_plot(stlabel, rdf, cprcp, curr_fillrate, curr_fillrate_cdf)
            row = pn.Row(f_prcp, f_totals)
        else:
            row = pn.pane.HTML(
                f"<center><h1>autocomplete={autocomplete_value}, no such station found</h1></center>",
                style={
                    'background-color': '#FFF6A0',
                    'color': '#A08040',
                    'border': '2px solid green',
                    'border-radius': '5px',
                    'padding': '10px'},
                width=800,
                height=150,
                sizing_mode='stretch_width'
            )
    else:
        row = pn.pane.HTML(
            f"<center><h1>autocomplete={autocomplete_value}</h1><h1>year={year_value}</h1></center>",
            style={
                'background-color': '#FFF6A0',
                'color': '#A08040',
                'border': '2px solid green',
                'border-radius': '5px',
                'padding': '10px'},
            width=800,
            height=150,
            sizing_mode='stretch_width'
        )
    return row


controls = pn.Row(autocomplete, input_year)
app = pn.Column(controls, p_drought_plot)
app.show()
app.servable()
