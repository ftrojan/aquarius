"""Worker asks for a station to calculate, calculates, saves to db and asks again, until gets input"""
import socket
import json
import logging
import utils


def get_response(request_dict: dict) -> dict:
    request = json.dumps(request_dict).encode('utf-8')
    logging.debug(f"sending request={request_dict}")
    s.sendall(request)
    response = s.recv(1024)
    response_dict = json.loads(response.decode('utf-8'))
    return response_dict


def dowork(station_id: str):
    prcp = utils.df_station(station_id)
    if not prcp.empty:
        refq = utils.calc_reference_quantiles(prcp)
        if not refq.empty:
            utils.insert_with_progress(refq, engine, table_name='reference', chunksize=2000)
    return


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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        todo = get_response({'command': 'get_station'})
        logging.debug(f'response={todo}')
        station = todo['station']
        if station:
            dowork(station)
            saved = get_response({'command': 'complete_station', 'station': station})
            logging.debug(f'response={saved}')
            stop = False
        else:
            stop = True
logging.debug("completed")
