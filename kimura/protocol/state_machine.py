from enum import Enum, auto
from typing import Optional
import logging
import os
import hashlib
import sys
from pathlib import Path
from transport.tcp import TCPTransport
from protocol.constants import ML_DSA_65_SIG_LEN, PROTOCOL_VERSION
# Fix imports for YOUR project structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")

from crypto.kdf import hkdf_sha256
from crypto.aead import AEADContext
from crypto.mlkem import MLKEM
from protocol.messages import (
    serialize_handshake_init, 
    parse_handshake_resp,
    serialize_handshake_resp,
    parse_handshake_init
)
from crypto.signing import (
    ensure_keys_exist, 
    load_mlkem_client_keys, load_mlkem_server_keys, 
    load_mldsa_client_keys, load_mldsa_server_keys,
    sign_message, verify_message, hash_file  
)

from file_transfer.transfer import (
    send_length_prefixed,
    recv_length_prefixed,
    send_file,
    recv_file
)
logger = logging.getLogger(__name__)

class TransferState(Enum):
    INIT = auto()
    HANDSHAKE_SENT = auto()
    HANDSHAKE_RECV = auto()
    HANDSHAKE_RESP_SENT = auto() 
    HANDSHAKE_COMPLETE = auto()
    FILE_TRANSFER = auto()
    TRANSFER_DONE = auto()
    ERROR = auto()

class ProtocolError(Exception):
    pass

class StateMachine:
    def __init__(self, key_path: str, role: str, signing_key: Optional[bytes] = None):
        self.state = TransferState.INIT
        self.role = role
        self.key_path = key_path
        self.sequence_number = 0
        self.transcript = hashlib.sha256()
        self.session_key: Optional[bytes] = None
        self.aead_ctx: Optional[AEADContext] = None
        self.kem: Optional[MLKEM] = None
        self.kem_public_key: Optional[bytes] = None
        self.kem_secret_key: Optional[bytes] = None
        self.ml_dsa_public_key: Optional[bytes] = None
        self.ml_dsa_secret_key: Optional[bytes] = None
        self.signing_key: Optional[bytes] = None
        self.peer_kem_public_key: Optional[bytes] = None  
        self.peer_ml_dsa_public_key: Optional[bytes] = None
        ensure_keys_exist(key_path, role)
        if role == "server":
            self.kem_public_key, self.kem_secret_key = load_mlkem_server_keys(key_path)
            self.ml_dsa_public_key, self.ml_dsa_secret_key = load_mldsa_server_keys(key_path)
        elif role == "client":
            self.kem_public_key, self.kem_secret_key = load_mlkem_client_keys(key_path)
            self.ml_dsa_public_key, self.ml_dsa_secret_key = load_mldsa_client_keys(key_path)
        else:
            raise ValueError("role must be 'server' or 'client'")
        # 3. Signing key
        self.signing_key = self.ml_dsa_secret_key if signing_key is None else signing_key
        # 4. Initialize KEM
        self.kem = MLKEM("ML-KEM-768")

    async def transition(self, event: str, reader=None, writer=None, **kwargs) -> None:
        logger.debug(f"[{self.role.upper()}] {self.state.name} → {event}")
        transitions = {
            TransferState.INIT: {
                "send_handshake": self._client_send_handshake,
                "recv_handshake": self._server_recv_handshake,
            },
            TransferState.HANDSHAKE_SENT: {
                "recv_response": self._client_recv_response,
            },
            TransferState.HANDSHAKE_RECV: {
                "send_response": self._server_send_response,
            },
            TransferState.HANDSHAKE_COMPLETE: {
                "start_send_file": self._start_send_file,
                "start_recv_file": self._start_recv_file,
            },
        }
        if self.state not in transitions or event not in transitions[self.state]:
            self._error(f"Invalid transition: {self.state.name} + {event}")
            return
        await transitions[self.state][event](reader, writer, **kwargs)

    async def _client_send_handshake(self, reader, writer, **kwargs):
        # 1. Build handshake body (this is what goes on the wire)
        body = (
            PROTOCOL_VERSION.to_bytes(2, "big") +
            self.kem_public_key +
            self.ml_dsa_public_key
        )

        # 2. Update transcript with what is being sent
        self.transcript.update(body)

        # 3. Sign the transcript hash
        signature = sign_message(
            self.transcript.digest(),
            self.ml_dsa_secret_key
        )

        # 4. Send
        full_message = body + signature
        await send_length_prefixed(writer, full_message)

        self.state = TransferState.HANDSHAKE_SENT
        logger.info(f"{self.role.upper()}: Handshake init sent")


    async def _server_recv_handshake(self, reader, writer, **kwargs):
        data = await recv_length_prefixed(reader)
        body = data[:-ML_DSA_65_SIG_LEN]
        signature = data[-ML_DSA_65_SIG_LEN:]
        # 1. Parse version
        version = int.from_bytes(body[:2], "big")
        if version != PROTOCOL_VERSION:
            self._error("Unsupported protocol version")
        # 2. Parse keys
        offset = 2
        client_kem_pk = body[offset:offset+len(self.kem_public_key)]
        offset += len(self.kem_public_key)
        client_dsa_pk = body[offset:offset+len(self.ml_dsa_public_key)]
        # 3. Update transcript BEFORE verify
        self.transcript.update(body)
        # 4. Verify transcript-bound signature
        if not verify_message(
            self.transcript.digest(),
            signature,
            client_dsa_pk
        ):
            self._error("Client signature invalid")
        self.peer_kem_public_key = client_kem_pk
        self.peer_ml_dsa_public_key = client_dsa_pk
        self.state = TransferState.HANDSHAKE_RECV
        logger.info(f"{self.role.upper()}: Client handshake verified")

    async def _client_recv_response(self, reader, writer, **kwargs):
        """CLIENT: Receive+VERIFY server's signed ciphertext."""
        data = await recv_length_prefixed(reader)
        resp_msg = data[:-ML_DSA_65_SIG_LEN]
        signature = data[-ML_DSA_65_SIG_LEN:]
        server_dsa_pk, _ = load_mldsa_server_keys(self.key_path)
        self.transcript.update(resp_msg)  # add received message to transcript
        if not verify_message(
            self.transcript.digest(),   # verify transcript hash
            signature,
            server_dsa_pk
        ):
            self._error("Server response signature invalid")
        ciphertext = parse_handshake_resp(resp_msg)
        shared_secret = self.kem.decaps(ciphertext, self.kem_secret_key)
        transcript_hash = self.transcript.digest()
        self.session_key = hkdf_sha256(shared_secret, b"", b"pqc_session_v1" + transcript_hash, 32)
        self.aead_ctx = AEADContext(self.session_key)
        self.peer_ml_dsa_public_key = server_dsa_pk
        self.state = TransferState.HANDSHAKE_COMPLETE
        await self.send_protected(writer, b"HANDSHAKE_OK")
        logger.info(f"{self.role.upper()}: Handshake complete")

    async def _server_send_response(self, reader, writer, **kwargs):
        """SERVER: Sign+send ciphertext using CLIENT's PK."""
        ciphertext, shared_secret = self.kem.encaps(self.peer_kem_public_key)
        resp_msg = serialize_handshake_resp(ciphertext)
        self.transcript.update(resp_msg)
        transcript_hash = self.transcript.digest()
        self.session_key = hkdf_sha256(
            shared_secret,
            b"",
            b"pqc_session_v1" + transcript_hash,
            32
        )
        self.aead_ctx = AEADContext(self.session_key)
        signature = sign_message(transcript_hash, self.ml_dsa_secret_key)
        await send_length_prefixed(writer, resp_msg + signature)
        self.state = TransferState.HANDSHAKE_RESP_SENT
        logger.info(f"{self.role.upper()}: Handshake response sent")
        self.state = TransferState.HANDSHAKE_COMPLETE
        ack = await self.recv_protected(reader)
        if ack != b"HANDSHAKE_OK":
            self._error("Client failed to verify handshake")
        logger.info(f"{self.role.upper()}: Handshake complete (client confirmed)")


    async def _start_send_file(self, reader, writer, filepath: str):
        """CLIENT: Send file_hash + signature + encrypted_file."""
        if not self.aead_ctx:
            self._error("No AEAD context")
        file_hash = hash_file(Path(filepath))
        signature = sign_message(file_hash, self.ml_dsa_secret_key)
        await self.send_protected(writer, file_hash + signature)
        await send_file(writer, Path(filepath), self.aead_ctx)
        logger.info(f"{self.role.upper()}: File transfer complete")

    async def _start_recv_file(self, reader, writer, output_path: str):
        """SERVER: Receive+VERIFY file_hash + signature + encrypted_file."""
        if not self.aead_ctx:
            self._error("No AEAD context")
        metadata = await self.recv_protected(reader)
        file_hash = metadata[:32]      # SHA256 digest
        signature = metadata[32:]      # ML-DSA signature
        if not verify_message(file_hash, signature, self.peer_ml_dsa_public_key):
            self._error("File signature invalid")
        # Receive encrypted file
        await recv_file(reader, writer, Path(output_path), self.aead_ctx)
        logger.info(f"{self.role.upper()}: File verified")

    def _error(self, reason: str):
        self.state = TransferState.ERROR
        raise ProtocolError(reason)

    def is_ready_for_transfer(self) -> bool:
        return self.state == TransferState.HANDSHAKE_COMPLETE and self.aead_ctx is not None
    
    def get_aead_context(self) -> AEADContext:
        if not self.aead_ctx:
            self._error("AEAD context not initialized")
        return self.aead_ctx
    async def send_protected(self, writer, payload: bytes):
        """NEW: ALL post-handshake sends go through this"""
        if self.state != TransferState.HANDSHAKE_COMPLETE or not self.aead_ctx:
            raise ProtocolError(f"State {self.state.name}, AEAD: {self.aead_ctx is not None}")
        seq = self.sequence_number.to_bytes(8, 'big')
        self.sequence_number += 1
        nonce = self.aead_ctx.generate_nonce()
        nonce = hashlib.sha256(nonce + seq).digest()[:12]  
        encrypted = self.aead_ctx.encrypt(payload, nonce)
        msg = seq + nonce + encrypted 
        await send_length_prefixed(writer, msg)

    async def recv_protected(self, reader) -> bytes:
        """ NEW: ALL post-handshake receives go through this"""
        if self.state != TransferState.HANDSHAKE_COMPLETE:
            raise ProtocolError("Handshake required")
        
        msg = await recv_length_prefixed(reader)
        seq = msg[:8]
        nonce = msg[8:20]  # 12 bytes
        encrypted = msg[20:]
        
        # STRICT sequence verification
        expected_seq = self.sequence_number.to_bytes(8, 'big')
        if seq != expected_seq:
            self._error(f"Replay! Expected {expected_seq.hex()}, got {seq.hex()}")
        
        payload = self.aead_ctx.decrypt(encrypted, nonce)
        self.sequence_number += 1
        return payload

    async def send_signed_data(self, writer, data: bytes):
        if self.state != TransferState.HANDSHAKE_COMPLETE:
            raise ProtocolError("Must complete handshake first")
        data_hash = hashlib.sha256(data).digest()  # 32 bytes
        signature = sign_message(data_hash, self.ml_dsa_secret_key)  # YOUR existing function!
        payload = data_hash + signature + data
        await self.send_protected(writer, payload)
        logger.info(f"{self.role.upper()}: Sent {len(data)/1024/1024:.1f}MB")

    async def recv_and_verify_data(self, reader) -> bytes:
        if self.state != TransferState.HANDSHAKE_COMPLETE:
            raise ProtocolError("Must complete handshake first")
        data = await self.recv_protected(reader)
        data_hash = data[:32]
        signature = data[32:32+ML_DSA_65_SIG_LEN]  # Your constant
        received_data = data[32+ML_DSA_65_SIG_LEN:]
        if not verify_message(data_hash, signature, self.peer_ml_dsa_public_key):
            raise ProtocolError("❌ Data signature invalid")
        if hashlib.sha256(received_data).digest() != data_hash:
            raise ProtocolError("❌ Hash mismatch")
        logger.info(f"{self.role.upper()}: Verified {len(received_data)/1024/1024:.1f}MB")
        return received_data

