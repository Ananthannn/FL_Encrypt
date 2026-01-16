import struct
import json
import pickle

# -------------------------
# LOW LEVEL
# -------------------------
def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data += packet
    return data


# -------------------------
# JSON
# -------------------------
def send_json(sock, obj):
    payload = json.dumps(obj).encode()
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def recv_json(sock):
    size = struct.unpack("!Q", recv_exact(sock, 8))[0]
    return json.loads(recv_exact(sock, size).decode())


# -------------------------
# PICKLE
# -------------------------
def send_pickle(sock, obj):
    payload = pickle.dumps(obj)
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def recv_pickle(sock):
    size = struct.unpack("!Q", sock.recv(8))[0]
    data = b""
    while len(data) < size:
        data += sock.recv(size - len(data))
    return pickle.loads(data)
