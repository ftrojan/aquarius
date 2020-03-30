"""Worker asks for a station to calculate, calculates, saves to db and asks again, until gets input"""
import socket
import json
import logging
import utils


def get_response(s, request_dict: dict) -> dict:
    request = json.dumps(request_dict).encode('utf-8')
    logging.debug(f"sending request={request_dict}")
    s.sendall(request)
    response = s.recv(1024)
    response_dict = json.loads(response.decode('utf-8'))
    return response_dict


def dowork(station_id: str):
    if station_id:
        prcp = utils.df_station(station_id)
        if not prcp.empty:
            refq = utils.calc_reference_quantiles(prcp)
            if not refq.empty:
                utils.insert_with_progress(refq, engine, table_name='reference', chunksize=2000)
    return


def get_station():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        todo = get_response(s, {'command': 'get_station'})
    logging.debug(f'response={todo}')
    todo_station = todo['station']
    return todo_station


def complete_station(station_id: str) -> bool:
    if station_id:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            saved = get_response(s, {'command': 'complete_station', 'station': station})
        logging.debug(f'response={saved}')
        stop_flag = False
    else:
        stop_flag = True
    return stop_flag


logging.basicConfig(
        level=logging.DEBUG,
        format=utils.logfmt,
        handlers=[logging.StreamHandler()],
    )
HOST = '127.0.0.1'
PORT = 50007
engine = utils.sql_engine()
stop = False
while not stop:
    station = get_station()
    dowork(station)
    stop = complete_station(station)
logging.debug("completed")
