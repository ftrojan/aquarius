"""
Flask autocomplete example.

From https://stackoverflow.com/questions/34704997/jquery-autocomplete-in-flask
"""
import logging
import pandas as pd
import constants
from flask import Flask, request, jsonify, render_template

logging.basicConfig(
    level=logging.DEBUG,
    format=constants.logfmt,
    handlers=[logging.StreamHandler()],
)
logging.debug("started")
dfc = pd.read_fwf('../../data/station/ghcnd-countries.txt')
dfc.columns = ['country_code', 'country_name']
logging.debug(f"{len(dfc)} countries loaded")
app = Flask(__name__)


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q')
    logging.debug(f"search={search}")
    flag_match = dfc.country_name.str.match(pat=search, case=False)
    logging.debug(f"{flag_match.sum()} countries match")
    results = dfc.country_name[flag_match].tolist()
    return jsonify(matching_results=results)


@app.route('/autocomplete_demo')
def autocomplete_demo():
    return render_template('demo.html')
