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
    sign_message, verify_message, verify_peer_key
)
from file_transfer.transfer import (
    send_length_prefixed,
    recv_length_prefixed
)
logger = logging.getLogger(__name__)
class TransferState(Enum):
    INIT = auto()
    HANDSHAKE_SENT = auto()
    HANDSHAKE_RECV = auto()
    HANDSHAKE_RESP_SENT = auto() 
    HANDSHAKE_COMPLETE = auto()
    ERROR = auto()

class ProtocolError(Exception):
    pass
class AEADPair:
    def __init__(self, send_ctx: AEADContext, recv_ctx: AEADContext, send_seq: int = 0, recv_seq: int = 0):
        self.send_ctx = send_ctx
        self.recv_ctx = recv_ctx
        self.send_seq = send_seq
        self.recv_seq = recv_seq

class StateMachine:
    def __init__(self, key_path: str, role: str, signing_key: Optional[bytes] = None):
        self.state = TransferState.INIT
        self.role = role
        self.key_path = key_path
        self.transcript = hashlib.sha256()
        self.session_key: Optional[bytes] = None
        self.aead_ctx: Optional[AEADPair] = None
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
                "send_data": self.send_protected,
                "recv_data": self.recv_protected,
                # FIXED FILE TRANSFER - pass kwargs through
                "start_send_file": self._send_file,
                "start_recv_file": self._recv_file,
            }
        }
        if self.state not in transitions or event not in transitions[self.state]:
            self.state = TransferState.ERROR
            self._error(f"Invalid transition: {self.state.name} + {event}")
            return
        await transitions[self.state][event](reader, writer, **kwargs)
        logger.info(f"[{self.role}] {self.state.name} --[{event}]--> Writer.active={writer.is_closing() if writer else 'no'}")


    async def _client_send_handshake(self, reader, writer, **kwargs):
        body = self.kem_public_key + self.ml_dsa_public_key
        domain = b"client handshake init"
        
        # 1. Sign body FIRST
        sig_input = hashlib.sha256(domain + body).digest()
        signature = sign_message(sig_input, self.ml_dsa_secret_key)
        
        # 2. Serialize COMPLETE message
        full_message = serialize_handshake_init(PROTOCOL_VERSION, self.kem_public_key, self.ml_dsa_public_key, signature)
        
        # 3. BOTH SIDES hash the FULL SENT MESSAGE
        self.transcript.update(full_message)
        await send_length_prefixed(writer, full_message)
        self.state = TransferState.HANDSHAKE_SENT

    async def _server_recv_handshake(self, reader, writer, **kwargs):
        data = await recv_length_prefixed(reader)
        self.transcript.update(data)  # Already correct
        
        client_kem_pk, client_dsa_pk, signature = parse_handshake_init(data)
        verify_peer_key(client_dsa_pk, self.key_path, "client")
        
        # Verify using BODY ONLY (matches client signing)
        body = client_kem_pk + client_dsa_pk  
        domain = b"client handshake init"
        if not verify_message(hashlib.sha256(domain + body).digest(), signature, client_dsa_pk):
            self._error("Client signature invalid")
        
        self.peer_kem_public_key = client_kem_pk
        self.peer_ml_dsa_public_key = client_dsa_pk
        self.state = TransferState.HANDSHAKE_RECV

    async def _server_send_response(self, reader, writer, **kwargs):
        ciphertext, shared_secret = self.kem.encaps(self.peer_kem_public_key)
        resp_msg = serialize_handshake_resp(ciphertext, self.ml_dsa_public_key)
        
        # CRITICAL: BOTH SIDES hash resp_msg ONLY for keys
        self.transcript.update(resp_msg)  # ← BEFORE signature!
        transcript_hash = self.transcript.digest()
        
        # Sign resp_msg only (like client did)
        domain = b"server handshake response"
        signature = sign_message(hashlib.sha256(domain + resp_msg).digest(), self.ml_dsa_secret_key)
        
        full_msg = resp_msg + signature
        await send_length_prefixed(writer, full_msg)
        
        # Derive keys using transcript_hash (SAME on both sides now!)
        client_key = hkdf_sha256(shared_secret, b"", b"client->server" + transcript_hash, 32)
        server_key = hkdf_sha256(shared_secret, b"", b"server->client" + transcript_hash, 32)
        send_ctx = AEADContext(server_key)
        recv_ctx = AEADContext(client_key)
        self.aead_ctx = AEADPair(send_ctx, recv_ctx)
        self.session_key = transcript_hash
        self.state = TransferState.HANDSHAKE_COMPLETE
        ack = await self.recv_protected(reader)
        if ack != b"HANDSHAKE_OK": self._error("No ack")

    async def _client_recv_response(self, reader, writer, **kwargs):
        data = await recv_length_prefixed(reader)
        resp_msg = data[:-ML_DSA_65_SIG_LEN]
        signature = data[-ML_DSA_65_SIG_LEN:]
        
        ciphertext, server_dsa_pk = parse_handshake_resp(resp_msg)
        verify_peer_key(server_dsa_pk, self.key_path, "server")
        
        domain = b"server handshake response"
        if not verify_message(hashlib.sha256(domain + resp_msg).digest(), signature, server_dsa_pk):
            self._error("Server sig invalid")
        
        # CRITICAL: Hash resp_msg ONLY (matches server)
        self.transcript.update(resp_msg)  # ← Already correct!
        shared_secret = self.kem.decaps(ciphertext, self.kem_secret_key)
        transcript_hash = self.transcript.digest()
        
        # SAME key derivation
        client_key = hkdf_sha256(shared_secret, b"", b"client->server" + transcript_hash, 32)
        server_key = hkdf_sha256(shared_secret, b"", b"server->client" + transcript_hash, 32)
        send_ctx = AEADContext(client_key)
        recv_ctx = AEADContext(server_key)
        self.aead_ctx = AEADPair(send_ctx, recv_ctx)
        self.session_key = transcript_hash
        self.peer_ml_dsa_public_key = server_dsa_pk
        self.state = TransferState.HANDSHAKE_COMPLETE
        await self.send_protected(writer, b"HANDSHAKE_OK")
        logger.info(f"{self.role.upper()}: Handshake complete")

    def _error(self, reason: str):
        self.state = TransferState.ERROR
        raise ProtocolError(reason)

    def is_ready_for_transfer(self) -> bool:
        return self.state == TransferState.HANDSHAKE_COMPLETE and self.aead_ctx is not None
    
    def get_aead_context(self) -> AEADPair:
        if not self.aead_ctx:
            self._error("AEAD context not initialized")
        return self.aead_ctx
    
    async def send_protected(self, reader, writer, payload: bytes):
        """ALL post-handshake sends go through this"""
        if self.state != TransferState.HANDSHAKE_COMPLETE or not self.aead_ctx:
            raise ProtocolError(f"State {self.state.name}, AEAD: {self.aead_ctx is not None}")
        seq = self.aead_ctx.send_seq.to_bytes(8, 'big')
        self.aead_ctx.send_seq += 1
        nonce = hkdf_sha256(seq, self.session_key , b"nonce", 12)
        encrypted = self.aead_ctx.send_ctx.encrypt(payload, nonce)
        msg = seq + nonce + encrypted 
        await send_length_prefixed(writer, msg)

    async def recv_protected(self, reader) -> bytes:
        """ALL post-handshake receives go through this"""
        if self.state != TransferState.HANDSHAKE_COMPLETE or not self.aead_ctx:
            raise ProtocolError("Handshake required")
        msg = await recv_length_prefixed(reader)
        seq = msg[:8]
        nonce = msg[8:20]  # 12 bytes
        encrypted = msg[20:]
        expected_seq = self.aead_ctx.recv_seq.to_bytes(8, 'big')
        if seq != expected_seq:
            self._error(f"Replay! Expected {expected_seq.hex()}, got {seq.hex()}")
        payload = self.aead_ctx.recv_ctx.decrypt(encrypted, nonce)
        self.aead_ctx.recv_seq += 1
        return payload
    async def _send_file(self, reader, writer, filepath: str):
        """Client: Send file post-handshake"""
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        filename = Path(filepath).name.encode()
        msg = (
            len(filename).to_bytes(4, 'big') +      # filename length
            filename +                              # filename
            len(file_data).to_bytes(8, 'big') +     # file size  
            file_data                               # file content
        )
        await self.send_protected(reader, writer, msg)

    async def _recv_file(self, reader, writer, output_path: str):
        """Server: Receive file post-handshake"""
        msg = await self.recv_protected(reader)
        
        filename_len = int.from_bytes(msg[:4], 'big')
        filename = msg[4:4+filename_len].decode()
        data_start = 4 + filename_len
        data_len = int.from_bytes(msg[data_start:data_start+8], 'big')
        file_data = msg[data_start+8:data_start+8+data_len]
        
        with open(output_path, 'wb') as f:
            f.write(file_data)
        logger.info(f"Received {filename} ({len(file_data)/1024/1024:.1f}MB)")

