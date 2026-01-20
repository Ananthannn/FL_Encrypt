# protocol/constants.py
"""PQC protocol constants."""
import os
import struct

PROTOCOL_VERSION = 1

HEADER_FORMAT = ">BBI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # automatically 6 bytes

# Crypto params
MLKEM_VARIANT = "ML-KEM-768"  # NIST standard
MLDSA_VARIANT = "ML-DSA-65"   # NIST standard  
SESSION_KEY_LEN = 32          # AES-256
NONCE_LEN = 12                # AES-GCM
ML_DSA_65_SIG_LEN = 3309    # Signature length for ML-DSA-65
# Protocol messages
MSG_HANDSHAKE_INIT = 0x01
MSG_HANDSHAKE_RESP = 0x02
MSG_FILE_START = 0x03
MSG_FILE_CHUNK = 0x04
MSG_FILE_DONE = 0x05
 
 
# Your liboqs-python KEMs
KEMS = {
    "ML-KEM-512": "ML-KEM-512",
    "ML-KEM-768": "ML-KEM-768", 
    "ML-KEM-1024": "ML-KEM-1024"
}
# Your liboqs-python DSAs
DSAS = {
    "ML-DSA-35": "ML-DSA-35",
    "ML-DSA-65": "ML-DSA-65",
    "ML-DSA-85": "ML-DSA-85"
}
# File transfer params
MAX_CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB
TRANSFER_TIMEOUT = 30   # seconds
# Misc
LOG_LEVEL = os.getenv("PQC_LOG_LEVEL", "INFO")
DEFAULT_PORT = 8443