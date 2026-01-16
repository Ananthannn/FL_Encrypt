import socket
import json
import pickle
import numpy as np


class FLServer:
    def __init__(
        self,
        model,
        model_meta,
        max_clients,
        ip="0.0.0.0",
        port=9999
    ):
        self.model = model
        self.model_meta = model_meta
        self.max_clients = max_clients
        self.ip = ip
        self.port = port

    # ------------------------------
    # Socket helpers
    # ------------------------------
    def _recv_all(self, sock, n_bytes):
        data = b""
        while len(data) < n_bytes:
            packet = sock.recv(n_bytes - len(data))
            if not packet:
                raise ConnectionError("Client disconnected")
            data += packet
        return data

    def _recv_pickle(self, sock):
        size = int.from_bytes(self._recv_all(sock, 8), "big")
        payload = self._recv_all(sock, size)
        return pickle.loads(payload)

    def _send_pickle(self, sock, obj):
        payload = pickle.dumps(obj)
        sock.sendall(len(payload).to_bytes(8, "big"))
        sock.sendall(payload)

    def _send_json(self, sock, obj):
        payload = json.dumps(obj).encode()
        sock.sendall(len(payload).to_bytes(8, "big"))
        sock.sendall(payload)

    # ------------------------------
    # FedAvg
    # ------------------------------
    def fedavg(self, client_weights, client_sizes):
        total = sum(client_sizes)
        new_weights = []

        for layer_weights in zip(*client_weights):
            weighted = [
                w * (n / total)
                for w, n in zip(layer_weights, client_sizes)
            ]
            new_weights.append(np.sum(weighted, axis=0))

        return new_weights

    # ------------------------------
    # One FL round
    # ------------------------------
    def run_round(self, client_sizes):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.ip, self.port))
        server.listen(self.max_clients)

        print(f"[SERVER] Listening on {self.ip}:{self.port}")
        print(f"[SERVER] Waiting for {self.max_clients} clients")

        clients = []

        while len(clients) < self.max_clients:
            client, addr = server.accept()
            client.settimeout(300)  # ⬅ longer timeout
            print(f"[SERVER] Client connected: {addr}")

            try:
                client.sendall(b"READY")
                if client.recv(16) == b"OK":
                    clients.append(client)
                else:
                    client.close()
            except Exception:
                client.close()

        print("[SERVER] All clients connected")

        # Send metadata + global weights
        for c in clients:
            self._send_json(c, self.model_meta)
            self._send_pickle(c, self.model.get_weights())

        print("[SERVER] Sent metadata and global weights")

        # Receive updates
        client_weights = []

        for i, c in enumerate(clients):
            try:
                print(f"[SERVER] Receiving update from client {i+1}")
                weights = self._recv_pickle(c)
                client_weights.append(weights)
            except Exception as e:
                print(f"[SERVER] Client {i+1} failed:", e)
            finally:
                c.close()

        server.close()

        if len(client_weights) != self.max_clients:
            raise RuntimeError("[SERVER] Not all client updates received")

        print("[SERVER] Aggregating (FedAvg)")
        new_weights = self.fedavg(client_weights, client_sizes)
        self.model.set_weights(new_weights)

        print("[SERVER] Global model updated")
        return new_weights
