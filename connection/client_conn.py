import socket
from connection.protocol import recv_json, recv_pickle, send_pickle


class FLClient:
    def __init__(self, server_ip="127.0.0.1", port=9999):
        self.server_ip = server_ip
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.port))

        if self.sock.recv(16) != b"READY":
            raise RuntimeError("Server not ready")

        self.sock.sendall(b"OK")
        print("✅ Connected to server")

    def receive_model_metadata(self):
        return recv_json(self.sock)

    def receive_global_weights(self):
        return recv_pickle(self.sock)

    def send_local_weights(self, weights):
        send_pickle(self.sock, weights)

    def close(self):
        if self.sock:
            self.sock.close()
