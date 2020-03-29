import socket
import logging
from time import sleep
import utils


class ClientObject:

    def __init__(self, host_address, server_port, port=0):
        self._server = (host_address, server_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host_address, port))

    def handshake(self):
        logging.debug('sending marco')
        self.socket.sendto('marco'.encode(encoding='utf-8'), self._server)
        sleep(.1)
        self.socket.sendto('marco'.encode(encoding='utf-8'), self._server)
        while True:
            if str(self.socket.recvfrom(1024)[0]) == 'polo':
                break
        # self._s.sendto('marco',self._server)
        # self._s.sendto('marco',self._server)
        logging.debug(' connection verified')
        self.socket.sendto('confirm'.encode(encoding='utf-8'), self._server)
        self.socket.setblocking(0)
        return True

    def recieve(self, mode = 0):
        _data, _addr = self.socket.recvfrom(1024)
        if mode == 0:
            return str(_data)
        if mode == 1:
            return int(_data)
        if mode == 2:
            return float(_data)
        if mode == 3:
            return tuple(_data)

    def send(self, data):
        self.socket.sendto(str(data).encode(encoding='utf-8'), self._server)

    def close(self):
        self.socket.close()
        logging.debug('_socket closed_')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format=utils.logfmt,
        handlers=[logging.StreamHandler()],
    )
    host = '127.0.0.1'
    talk = ClientObject(host, server_port=6003, port=6004)
    talk.handshake()
    # while True:
        # print talk.recieve()
