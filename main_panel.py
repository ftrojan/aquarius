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
    max_stations = 200000
    agg_stations = 100
    if autocomplete_value and year:
        num_stations = tree.at[autocomplete_value, 'num_stations']
        stid = node_station.loc[[autocomplete_value], :]
        df = stid.set_index('station').join(drought, how='inner')
        plotdf = utils.drought_add_facecolor(df)
        bokeh_map = utils.drought_map(plotdf)
        if num_stations == 1:
            station = stid.station[0]
            stlabel = utils.station_label(stations.loc[station, :])
            logging.debug(f"calling drought_rate_data with stid={station} and year={year}")
            rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf = \
                utils.drought_rate_data(station, year, engine=engine)
            f_prcp = utils.cum_prcp_plot(stlabel, rdf, cprcp, curr_drought_rate)
            b_prcp = pn.pane.Bokeh(f_prcp)
            dft = totals.loc[totals['station'] == station, :]
            f_totals = utils.totals_barchart(dft)
            # utils.cum_fillrate_plot(stlabel, rdf, cprcp, curr_fillrate, curr_fillrate_cdf)
            row1 = pn.Row(b_prcp, f_totals)
            row = pn.Column(row1, pn.pane.Bokeh(bokeh_map))
        else:  # num_stations > 1
            if num_stations <= max_stations:
                p = utils.drought_rate_plot(plotdf)
            else:
                p = utils.drought_rate_plot_agg(plotdf, agg_stations)
            row = pn.Column(pn.pane.Bokeh(p), pn.pane.Bokeh(bokeh_map))
    else:  # empty autocomplete or not valid year - assume world at current_year
        plotdf = utils.drought_add_facecolor(drought)
        bokeh_map = utils.drought_map(plotdf)
        p = utils.drought_rate_plot(drought)
        row = pn.Column(pn.pane.Bokeh(p), pn.pane.Bokeh(bokeh_map))
    return row


@pn.depends(autocomplete.param.value)
def p_id(autocomplete_value: str):
    if autocomplete_value:
        num_stations = tree.at[autocomplete_value, 'num_stations']
        if num_stations == 1:
            stid = node_station.loc[autocomplete_value, 'station']
            result = pn.pane.HTML(
                stid,
                style={
                    'background-color': '#FFF6A0',
                    'color': '#A08040',
                    'border': '2px solid green',
                    'border-radius': '5px',
                    'padding': '10px',
                },
            )
        else:  # num_stations > 1
            result = None
    else:  # empty autocomplete or not valid year - assume world at current_year
        result = pn.pane.HTML(
            'Source: <a href="https://www.ncdc.noaa.gov/ghcn-daily-description">GHCN</a>',
            style={
                'background-color': '#FFF6A0',
                'color': '#A08040',
                'border': '2px solid green',
                'border-radius': '5px',
                'padding': '10px',
            },
        )
    return result


controls = pn.Row(autocomplete, input_year)
app = pn.Column(controls, p_drought_plot, p_id)
app.show()
app.servable()
