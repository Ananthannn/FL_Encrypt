import socket
import json
import pickle
import struct


class FLClient:
    def __init__(self, server_ip="127.0.0.1", port=9999):
        self.server_ip = server_ip
        self.port = port
        self.sock = None

    # ------------------------------
    # Socket helpers
    # ------------------------------
    def _recv_exact(self, n):
        data = b""
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Socket closed")
            data += packet
        return data

    def _recv_json(self):
        size = struct.unpack("!Q", self._recv_exact(8))[0]
        return json.loads(self._recv_exact(size).decode())

    def _recv_pickle(self):
        size = struct.unpack("!Q", self._recv_exact(8))[0]
        return pickle.loads(self._recv_exact(size))

    def _send_pickle(self, obj):
        payload = pickle.dumps(obj)
        self.sock.sendall(struct.pack("!Q", len(payload)))
        self.sock.sendall(payload)

    # ------------------------------
    # Protocol
    # ------------------------------
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.port))

        if self.sock.recv(16) != b"READY":
            raise RuntimeError("Server not ready")

        self.sock.sendall(b"OK")
        print("✅ Connected to server")

    def receive_model_metadata(self):
        return self._recv_json()

    def receive_global_weights(self):
        return self._recv_pickle()

    def send_local_weights(self, weights):
        self._send_pickle(weights)

    def close(self):
        if self.sock:
            self.sock.close()
