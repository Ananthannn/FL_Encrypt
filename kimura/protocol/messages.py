# protocol/messages.py - 100% YOUR CRYPTO
"""PQC protocol using YOUR crypto/mlkem.py + crypto/kdf.py."""

import struct
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypto.mlkem import MLKEM           
from crypto.kdf import hkdf_sha256      
from crypto.aead import AEADContext    
from protocol.constants import (
    MSG_HANDSHAKE_INIT,
    MSG_HANDSHAKE_RESP,
    MSG_FILE_CHUNK,
    PROTOCOL_VERSION,
    NONCE_LEN,
    HEADER_FORMAT,
    HEADER_SIZE
)

def serialize_handshake_init(kem_pk: bytes, client_sig: bytes) -> bytes:  # ADD SIG PARAM
    kem_id = 1
    pk_len = len(kem_pk)
    sig_len = len(client_sig)  # ML-DSA-65 = 2420 bytes typically
    return (struct.pack('>BHHH', MSG_HANDSHAKE_INIT, kem_id, pk_len, sig_len)  # 1+2+2+2
            + kem_pk + client_sig)

def parse_handshake_init(data: bytes) -> tuple[bytes, bytes]:
    msg_type, kem_id, pk_len, sig_len = struct.unpack('>BHHH', data[:7])
    pk_end = 7 + pk_len
    return data[7:pk_end], data[pk_end:pk_end+sig_len]  # RETURNS (PK, SIG) ✓

def parse_handshake_init(data: bytes) -> bytes:
    """Extract KEM public key."""
    msg_type, kem_id, pk_len = struct.unpack('>BHH', data[:5])
    return data[5:5+pk_len]

def serialize_handshake_resp(ciphertext: bytes) -> bytes:
    """Server→Client: [1B type][2B kem_id][2B ct_len][ciphertext]"""
    kem_id = 1
    ct_len = len(ciphertext)
    return struct.pack('>BHH', MSG_HANDSHAKE_RESP, kem_id, ct_len) + ciphertext

def parse_handshake_resp(data: bytes) -> bytes:
    """Extract KEM ciphertext."""
    msg_type, kem_id, ct_len = struct.unpack('>BHH', data[:5])
    return data[5:5+ct_len]

def derive_session_key(shared_secret: bytes, salt: bytes = b"") -> bytes:
    """YOUR crypto/kdf.py HKDF!"""
    return hkdf_sha256(
        secret=shared_secret,
        salt=salt, 
        info=b"pqc_file_transfer_session",
        length=32  # AES-256
    )

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

