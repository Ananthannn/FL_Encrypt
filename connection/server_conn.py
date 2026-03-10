import socket
import numpy as np
from connection.protocol import send_json, send_pickle, recv_pickle


class FLServer:
    def __init__(self, model, model_meta, max_clients, ip="0.0.0.0", port=9999):
        self.model = model
        self.model_meta = model_meta
        self.max_clients = max_clients
        self.ip = ip
        self.port = port

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
    # One FL Round
    # ------------------------------
    def run_round(self, client_sizes):

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.ip, self.port))
        server.listen(self.max_clients)

        print(f"[SERVER] Listening on {self.ip}:{self.port}")
        clients = []

        while len(clients) < self.max_clients:
            client, addr = server.accept()
            print(f"[SERVER] Client connected: {addr}")

            client.sendall(b"READY")
            if client.recv(16) == b"OK":
                clients.append(client)
            else:
                client.close()

        print("[SERVER] All clients connected")

        # Send metadata and global weights
        for c in clients:
            send_json(c, self.model_meta)
            send_pickle(c, self.model.get_weights())

        # Receive client updates
        client_weights = []

        for i, c in enumerate(clients):
            print(f"[SERVER] Receiving update from client {i+1}")
            weights = recv_pickle(c)
            client_weights.append(weights)
            c.close()

        server.close()

        print("[SERVER] Aggregating (FedAvg)")
        new_weights = self.fedavg(client_weights, client_sizes)
        self.model.set_weights(new_weights)

        print("[SERVER] Global model updated")

        return new_weights
