# protocol/messages.py - 100% YOUR CRYPTO
"""PQC protocol using YOUR crypto/mlkem.py + crypto/kdf.py."""

import struct
import sys
import os
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kimura.crypto.mlkem import MLKEM           
from kimura.crypto.kdf import hkdf_sha256      
from kimura.crypto.aead import AEADContext    
from kimura.protocol.constants import (
    MSG_HANDSHAKE_INIT,
    MSG_HANDSHAKE_RESP,
    MSG_FILE_CHUNK,
    PROTOCOL_VERSION,
    NONCE_LEN,
    HEADER_FORMAT,
    HEADER_SIZE
)
ML_KEM_LEN = 1184   # 96 bytes
ML_DSA_LEN = 1952         # bytes
ML_DSA_SIG_LEN = 3309     # bytes
import struct
from kimura.protocol.constants import MSG_HANDSHAKE_INIT  # Add this constant

def serialize_handshake_init(version: int, kem_pk: bytes, dsa_pk: bytes, signature: bytes) -> bytes:
    # Header: msg_type + kem_len + dsa_len + sig_len
    kem_len = len(kem_pk)
    dsa_len = len(dsa_pk)
    sig_len = len(signature)
    header = struct.pack('>BHHH', MSG_HANDSHAKE_INIT, kem_len, dsa_len, sig_len)
    return header + kem_pk + dsa_pk + signature

def serialize_handshake_resp(ciphertext: bytes, dsa_pk: bytes) -> bytes:
    kem_id = 1
    ct_len = len(ciphertext)
    dsa_len = len(dsa_pk)
    header = struct.pack('>BHHH', MSG_HANDSHAKE_RESP, kem_id, ct_len, dsa_len)
    return header + ciphertext + dsa_pk

def parse_handshake_init(data: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Parse handshake init message WITH HEADER.
    Format:
    - Header: 1B msg_type + 2B kem_len + 2B dsa_len + 2B sig_len
    - KEM_pub: kem_len bytes
    - DSA_pub: dsa_len bytes
    - signature: sig_len bytes
    """
    if len(data) < 7:
        raise ValueError("Handshake init too short for header")

    # Unpack header
    msg_type, kem_len, dsa_len, sig_len = struct.unpack(">BHHH", data[:7])

    # Check remaining length
    if len(data) < 7 + kem_len + dsa_len + sig_len:
        raise ValueError("Handshake init too short for payload")

    kem_pk = data[7:7+kem_len]
    dsa_pk = data[7+kem_len:7+kem_len+dsa_len]
    signature = data[7+kem_len+dsa_len:7+kem_len+dsa_len+sig_len]

    # Sanity checks
    assert len(kem_pk) == kem_len
    assert len(dsa_pk) == dsa_len
    assert len(signature) == sig_len

    return kem_pk, dsa_pk, signature



def parse_handshake_resp(data: bytes) -> Tuple[bytes, bytes]:
    if len(data) < 7:
        raise ValueError("Too short")
    msg_type, kem_id, ct_len, dsa_len = struct.unpack('>BHHH', data[:7])  # Match serialize!
    ct = data[7:7+ct_len]
    dsa_pk = data[7+ct_len:7+ct_len+dsa_len]
    return ct, dsa_pk


def serialize_file_chunk(chunk_data: bytes, aead_ctx: AEADContext) -> bytes:
    """
    Encrypt & serialize a file chunk with version + nonce.
    Returns bytes ready for sending over TCP.
    """
    nonce = aead_ctx.generate_nonce()
    ciphertext = aead_ctx.encrypt(chunk_data, nonce)
    chunk_len = len(ciphertext)
    header = struct.pack(HEADER_FORMAT, PROTOCOL_VERSION, MSG_FILE_CHUNK, chunk_len)
    return header + nonce + ciphertext


def parse_file_chunk(data: bytes, aead_ctx: AEADContext) -> bytes:
    """
    Deserialize & decrypt a received file chunk.
    Returns plaintext bytes.
    """
    header = data[:HEADER_SIZE]
    version, msg_type, chunk_len = struct.unpack(HEADER_FORMAT, header)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version {version}")
    if msg_type != MSG_FILE_CHUNK:
        raise ValueError(f"Unexpected message type {msg_type}")
    nonce = data[HEADER_SIZE:HEADER_SIZE+NONCE_LEN]
    ciphertext = data[HEADER_SIZE+NONCE_LEN:HEADER_SIZE+NONCE_LEN+chunk_len]
    return aead_ctx.decrypt(ciphertext, nonce)
