import socket
import logging
from time import sleep
import utils


class ServerObject:

    def __init__(self, host_address, port):
        self.host = host_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, port))
        self.address = None

    def handshake(self):
        logging.debug("Dispatcher Started. Awaiting marco")
        while True:
            data, address = self.socket.recvfrom(1024)
            if str(self.socket.recvfrom(1024)[0]) == 'marco':
                break
        logging.debug('marco recieved. sending polo...')
        while True:
            self.socket.sendto(data='polo'.encode(encoding='utf-8'), address=address)
            if str(self.socket.recvfrom(1024)[0]) == 'confirm':
                break
            sleep(.5)

        logging.debug('connection verified')
        self.address = address
        return True

    def send(self, data):
        self.socket.sendto(data=str(data).encode(encoding='utf-8'), address=self.address)

    def recieve(self, mode=0):
        _data, _addr = self.socket.recvfrom(1024)
        if mode == 0:
            return str(_data)
        if mode == 1:
            return int(_data)
        if mode == 2:
            return float(_data)
        if mode == 3:
            return tuple(_data)

    def change_port(self, port):
        self.socket.bind((self.host, port))

    def close(self):
        self.socket.close()
        logging.debug('socket closed')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format=utils.logfmt,
        handlers=[logging.StreamHandler()],
    )
    host = '127.0.0.1'
    talk = ServerObject(host, port=6003)
    talk.handshake()
