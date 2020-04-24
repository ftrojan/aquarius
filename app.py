"""Flask root app."""
# inspired by https://github.com/realpython/flask-bokeh-example/blob/master/tutorial.md

import logging
import pandas as pd
import utils
import constants
from dfply import X, mask
from flask import Flask, render_template, url_for
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.models import OpenURL, TapTool

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler()],
)
app = Flask(__name__)
js_resources = INLINE.render_js()
css_resources = INLINE.render_css()
logging.debug("logging works")
current_year = utils.get_current_year()
stations = utils.load_stations()
totals = pd.read_csv('../../data/yearly_totals/prcp_totals.csv')
logging.debug(f"{len(totals)} totals loaded")
engine = utils.sql_engine()
drought = utils.get_drought(engine).join(stations)
logging.debug(f"{len(drought)} stations with drought information loaded from {engine}")


@app.route('/')
def index():
    cz = drought >> mask(X.country_name == 'Czech Republic')
    plotdf = utils.drought_add_facecolor(cz)
    f_drought = utils.drought_rate_plot(plotdf)
    url = "/station_detail/station_id=@station"
    # taptool = f_drought.select(type=TapTool)
    # taptool.callback = OpenURL(url=url)
    f_drought.add_tools(TapTool(callback=OpenURL(url=url)))
    bokeh_map = utils.drought_map(plotdf)
    bokeh_map.add_tools(TapTool(callback=OpenURL(url=url)))
    script_drought, div_drought = components(f_drought)
    script_map, div_map = components(bokeh_map)
    return render_template(
        'main_page.html',
        stations=cz,
        js_resources=js_resources,
        css_resources=css_resources,
        script_drought=script_drought,
        script_map=script_map,
        div_drought=div_drought,
        div_map=div_map,
    )


@app.route('/station_detail/station_id=<station_id>')
def station_detail(station_id: str):
    url_back = url_for('index')
    logging.debug(f"station_id: {station_id}")
    station = stations.loc[station_id, :]
    stlabel = utils.station_label(station)
    rdf, cprcp, curr_drought_rate, curr_fillrate, curr_fillrate_cdf = \
        utils.drought_rate_data(station_id, current_year, engine=engine)
    dft = totals.loc[totals['station'] == station_id, :]
    f_prcp = utils.cum_prcp_plot(stlabel, rdf, cprcp, curr_drought_rate)
    f_totals = utils.totals_barchart(dft)
    script, div = components(f_prcp)
    script_totals, div_totals = components(f_totals)
    html = render_template(
        'station_detail.html',
        js_resources=js_resources,
        css_resources=css_resources,
        plot_script=script,
        plot_script_totals=script_totals,
        plot_div=div,
        plot_totals=div_totals,
        station=station,
        url_back=url_back)
    return html
