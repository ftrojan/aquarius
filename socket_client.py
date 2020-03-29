# Echo client program (https://docs.python.org/3/library/socket.html)
import socket
import json
import logging
import utils

logging.basicConfig(
        level=logging.DEBUG,
        format=utils.logfmt,
        handlers=[logging.StreamHandler()],
    )
HOST = '127.0.0.1'       # The host
PORT = 50007              # The same port as used by the server
request_dict = {
    'id': 123,
    'payload': 'Hello world'
}
request = json.dumps(request_dict).encode('utf-8')
logging.debug(f"sending request={request_dict}")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(request)
    response = s.recv(1024)
    response_dict = json.loads(response.decode('utf-8'))
logging.debug(f'response={response_dict}')
