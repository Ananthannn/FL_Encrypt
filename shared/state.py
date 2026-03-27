from enum import Enum

class WorkerState(Enum):
    CONNECTED = "CONNECTED"                 # Handshake done
    READY = "READY"                         # Idle, no model loaded
    
    WAITING_FOR_ROUND = "WAITING_FOR_ROUND" # Model loaded, waiting START_ROUND
    TRAINING = "TRAINING"                   # Actively training
    WAITING_FOR_AGGREGATE = "WAITING_FOR_AGGREGATE"  # Update sent

    FAILED = "FAILED"

class MasterState(Enum):
    IDLE = "IDLE"
    WAITING_FOR_WORKERS = "WAITING_FOR_WORKERS"
    ROUND_ACTIVE = "ROUND_ACTIVE"
    AGGREGATING = "AGGREGATING"
    BROADCASTING = "BROADCASTING"
