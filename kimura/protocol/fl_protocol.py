from enum import Enum
import struct

class FLMessageType(Enum):
    MODEL_FILE = 1
    MODEL_LOADED = 2
    TRAIN_CONFIG = 3
    START_TRAIN = 4
    UPDATE = 5
    AGGREGATED_MODEL = 6
    SHUTDOWN = 7


def serialize_fl_message(msg_type: FLMessageType, payload: bytes) -> bytes:
    header = struct.pack(">BI", msg_type.value, len(payload))
    return header + payload


def parse_fl_message(data: bytes):
    msg_type_val, length = struct.unpack(">BI", data[:5])
    payload = data[5:5+length]
    return FLMessageType(msg_type_val), payload

