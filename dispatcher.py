"""
Dispatcher responds with station numbers not calculated yet until no such remain.

It is for calculation of the reference table.
"""
import pandas as pd
import socket
import json
import logging
from datetime import datetime
import utils
import constants


def send_response(response_dict: dict):
    logging.debug(f"sending response={response_dict}")
    conn.sendall(json.dumps(response_dict).encode('utf-8'))


def get_station(job: pd.DataFrame) -> tuple:
    stations_available = job.index[job['dispatched_at'].isnull()]
    if len(stations_available) > 0:
        station_todo = stations_available[0]
        job.at[station_todo, 'dispatched_at'] = datetime.utcnow()
        sql_update = (
            "update reference_job\n"
            "set dispatched_at = now()\n"
            f"where station='{station_todo}'"
        )
        engine.execute(sql_update)
    else:
        station_todo = None
    return job, station_todo


def complete_station(job: pd.DataFrame, station_completed: str) -> pd.DataFrame:
    job.at[station_completed, 'completed_at'] = datetime.utcnow()
    sql_update = (
        "update reference_job\n"
        "set completed_at = now()\n"
        f"where station='{station_completed}'"
    )
    engine.execute(sql_update)
    return job


def get_perimeter(stdf: pd.DataFrame) -> float:
    perimeter = stdf.perimeter_km[stdf.dispatched_at.isnull()].min()
    return perimeter


logging.basicConfig(
    level=logging.INFO,
    format=constants.logfmt,
    handlers=[logging.StreamHandler()],
)
HOST = ''
PORT = 50007
engine = utils.sql_engine()
stations = utils.load_stations()
logging.info("calculating stations TODO with perimeter")
stations_todo = utils.get_stations_noref(engine, stations)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    pm = get_perimeter(stations_todo)
    logging.info(f"server listening on {HOST} {PORT}, {len(stations_todo)} stations todo, perimeter={pm:.1f}km.")
    while stations_todo['dispatched_at'].isnull().sum() > 0:  # after one request is served, listen for another one
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            logging.debug(f"Connected by {addr}")
            while True:  # until the socket is terminated by the worker
                num_todo = stations_todo['dispatched_at'].isnull().sum()
                logging.debug(f"{num_todo} stations todo")
                reqdata = conn.recv(1024)
                if not reqdata:
                    break
                request = json.loads(reqdata.decode('utf-8'))
                logging.debug(f"received request={request}")
                command = request['command']
                if command == 'get_station':
                    stations_todo, station = get_station(stations_todo)
                    response = {'station': station}
                    send_response(response)
                elif command == 'complete_station':
                    station = request['station']
                    stations_todo = complete_station(stations_todo, station)
                    logging.info(f"{num_todo} stations todo, perimeter={get_perimeter(stations_todo):.1f}km.")
                    response = {'station': station, 'completed': True}
                    send_response(response)
