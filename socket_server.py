# Echo server program (https://docs.python.org/3/library/socket.html)
import socket
import json
import logging
import utils

logging.basicConfig(
        level=logging.DEBUG,
        format=utils.logfmt,
        handlers=[logging.StreamHandler()],
    )
HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 50007              # Arbitrary non-privileged port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    logging.debug(f"server listening on {HOST} {PORT}")
    while True:  # after one request is served, listen for another one
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            logging.debug(f"Connected by {addr}")
            while True:
                request = conn.recv(1024)
                if not request:
                    break
                request_dict = json.loads(request.decode('utf-8'))
                logging.debug(f"received request={request_dict}")
                response = {
                    'header': 'ahoj',
                    'body': 'neco',
                }
                logging.debug(f"sending response={response}")
                conn.sendall(json.dumps(response).encode('utf-8'))
