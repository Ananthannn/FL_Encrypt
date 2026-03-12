from enum import Enum, auto
from typing import Optional
import logging
import os
import hashlib
import sys
from pathlib import Path
from kimura.protocol.constants import ML_DSA_65_SIG_LEN, PROTOCOL_VERSION
from kimura.file_transfer.transfer import chunked_send_file, recv_file
# Fix imports for YOUR project structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
from kimura.crypto.kdf import hkdf_sha256
from crypto.aead import AEADContext
from kimura.crypto.mlkem import MLKEM
from kimura.protocol.messages import (
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
        self.handshake_done = False
        self.peer_kem_public_key: Optional[bytes] = None  
        self.writer_active = False  # tracks if handshake completed and writer ready
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
        assert self.ml_dsa_public_key != self.ml_dsa_secret_key, \
            "❌ DSA public and secret keys are identical! This should never happen."
        # 4. Initialize KEM
        self.kem = MLKEM("ML-KEM-768")
    def get_peer_identity_key(self) -> bytes:
        """
        Returns the verified peer ML-DSA public key.
        Safe ONLY after handshake completes.
        """
        if self.peer_ml_dsa_public_key is None:
            raise RuntimeError("Peer identity key requested before handshake completion")
        return self.peer_ml_dsa_public_key

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
        logger.info(f"[{self.role}] {self.state.name} --[{event}]--> writer_active={self.writer_active}")

    async def _client_send_handshake(self, reader, writer, **kwargs):
        if self.state != TransferState.INIT:
            logger.warning(f"Handshake already sent or complete (state={self.state.name}), skipping")
            return
        # 1️⃣ Load raw PEM bytes of client DSA
        raw_client_dsa_pk, raw_client_dsa_sk = load_mldsa_client_keys(self.key_path)
        #logger.error(f"CLIENT ACTUAL DSA KEY LEN = {len(raw_client_dsa_pk)}")

        # 2️⃣ Build body and sign
        body = self.kem_public_key + raw_client_dsa_pk
        domain = b"client handshake init"
        sig_input = hashlib.sha256(domain + body).digest()
        signature = sign_message(sig_input, self.ml_dsa_secret_key)

        # 3️⃣ Serialize handshake init
        full_message = serialize_handshake_init(
            PROTOCOL_VERSION,
            self.kem_public_key,
            raw_client_dsa_pk,
            signature
        )
        # 3️⃣.5️⃣ Debug before sending
        #logger.info(f"[CLIENT] Sending handshake, KEM_pub={self.kem_public_key.hex()[:16]} DSA_pub={raw_client_dsa_pk.hex()[:16]}")
        #logger.info(f"[CLIENT] Handshake message len: {len(full_message)} bytes")
        # 4️⃣ Update transcript
        self.transcript.update(full_message)
        #print(f"[CLIENT] transcript after init: {self.transcript.hexdigest()}")

        # 5️⃣ Send handshake
        await send_length_prefixed(writer, full_message)
        await writer.drain()
        self.state = TransferState.HANDSHAKE_SENT

        # 6️⃣ Wait for server response
        response = await recv_length_prefixed(reader)
        # 6️⃣.5️⃣ Debug received server response
        logger.info(f"[CLIENT] Received response, {len(response)} bytes")
        # 7️⃣ Split response into message + signature
        resp_msg = response[:-ML_DSA_65_SIG_LEN]    # The raw server message
        signature = response[-ML_DSA_65_SIG_LEN:]   # Server's signature

        # 8️⃣ Parse server message
        ciphertext, server_dsa_pk = parse_handshake_resp(resp_msg)
        #logger.info(f"[CLIENT] Server DSA key (received): {server_dsa_pk.hex()[:16]}")
        #logger.info(f"[CLIENT] Server ciphertext: {ciphertext.hex()[:16]}")
        # 🔟 Verify server signature
        domain = b"server handshake response"
        if not verify_message(hashlib.sha256(domain + resp_msg).digest(), signature, server_dsa_pk):
            self._error("Server signature invalid")

        # 1️⃣1️⃣ Update transcript and derive session keys
        self.transcript.update(resp_msg)
        #print(f"[CLIENT] transcript before keys: {self.transcript.hexdigest()}")
        shared_secret = self.kem.decaps(ciphertext, self.kem_secret_key)
        transcript_hash = self.transcript.digest()
        client_key = hkdf_sha256(shared_secret, b"", b"client->server" + transcript_hash, 32)
        server_key = hkdf_sha256(shared_secret, b"", b"server->client" + transcript_hash, 32)
        send_ctx = AEADContext(client_key)
        recv_ctx = AEADContext(server_key)
        self.aead_ctx = AEADPair(send_ctx, recv_ctx)
        self.session_key = transcript_hash
        self.peer_ml_dsa_public_key = server_dsa_pk
        self.state = TransferState.HANDSHAKE_COMPLETE
        self.writer_active=True
        self.handshake_done = True
        #logger.info(f"[CLIENT] Derived AEAD keys, session_key: {self.session_key.hex()[:16]}")
        #logger.info(f"[CLIENT] AEAD send_seq={self.aead_ctx.send_seq}, recv_seq={self.aead_ctx.recv_seq}")

        # 5️⃣.5️⃣ Debug after sending handshake
        #logger.info(f"[CLIENT] Handshake sent. Waiting for server response...")
        # 1️⃣2️⃣ Send handshake ACK to server
        await self.send_protected(reader, writer, b"HANDSHAKE_OK")
        # 1️⃣3️⃣ Close writer safely
        #writer.close()
        #await writer.wait_closed()
        logger.info(f"{self.role.upper()}: Handshake complete")



    async def _server_recv_handshake(self, reader, writer, **kwargs):
        data = await recv_length_prefixed(reader)
        logger.info(f"[SERVER] Received handshake init, {len(data)} bytes")
        #logger.info(f"[SERVER] Raw handshake data (start): {data[:32].hex()}")

        self.transcript.update(data)
        #print(f"[SERVER] transcript after init: {self.transcript.hexdigest()}")

        # get the raw slices from handshake
        client_kem_pk, raw_client_dsa_bytes, signature = parse_handshake_init(data)
        #logger.error(f"CLIENT DSA KEY LENGTH = {len(raw_client_dsa_bytes)} bytes")
        #logger.info(f"[SERVER] Parsed client KEM_pub: {client_kem_pk.hex()[:16]}")
        #logger.info(f"[SERVER] Parsed client DSA_pub: {raw_client_dsa_bytes.hex()[:16]}")
        #logger.info(f"[SERVER] Parsed client signature (start): {signature.hex()[:16]}")

        # ✅ Use RAW bytes for TOFU
        verify_peer_key(raw_client_dsa_bytes, self.key_path, "client", ephemeral=True)

        ##logger.info(f"[SERVER] TOFU verification passed for client DSA key")
        # ✅ Verify signature using the same raw bytes
        body = client_kem_pk + raw_client_dsa_bytes
        domain = b"client handshake init"
        if not verify_message(hashlib.sha256(domain + body).digest(), signature, raw_client_dsa_bytes):
            self._error("Client signature invalid")
        # After verifying client signature and updating state
        #logger.info(f"[SERVER] Client handshake signature verified successfully")
        self.peer_kem_public_key = client_kem_pk
        self.peer_ml_dsa_public_key = raw_client_dsa_bytes
        self.state = TransferState.HANDSHAKE_RECV
        #logger.info(f"[SERVER] Handshake response sent, state updated to {self.state.name}")
        #logger.info(f"[SERVER] Stored client KEM_pub: {self.peer_kem_public_key.hex()[:16]}")
        #logger.info(f"[SERVER] Stored client DSA_pub: {self.peer_ml_dsa_public_key.hex()[:16]}")



    async def _server_send_response(self, reader, writer, **kwargs):
        ciphertext, shared_secret = self.kem.encaps(self.peer_kem_public_key)
        #logger.info(f"[SERVER] Encapsulated KEM, ciphertext (start): {ciphertext.hex()[:16]}")
        #logger.info(f"[SERVER] Shared secret derived (start): {shared_secret.hex()[:16]}")

        resp_msg = serialize_handshake_resp(ciphertext, self.ml_dsa_public_key)
        #logger.info(f"[SERVER] Serialized handshake response, len={len(resp_msg)} bytes")
        #logger.info(f"[SERVER] Server DSA key (used in resp): {self.ml_dsa_public_key.hex()[:16]}")

        # CRITICAL: BOTH SIDES hash resp_msg ONLY for keys
        self.transcript.update(resp_msg)  # ← BEFORE signature!
        #print(f"[SERVER] transcript before keys: {self.transcript.hexdigest()}")
        #logger.info(f"[SERVER] Transcript updated with resp_msg, digest: {self.transcript.hexdigest()}")
        transcript_hash = self.transcript.digest()
        
        # Sign resp_msg only (like client did)
        domain = b"server handshake response"
        signature = sign_message(hashlib.sha256(domain + resp_msg).digest(), self.ml_dsa_secret_key)
        #logger.info(f"[SERVER] Response signature created, start: {signature.hex()[:16]}")
        full_msg = resp_msg + signature
        logger.info(f"[SERVER] Sending full handshake response, len={len(full_msg)} bytes")
        #logger.info(f"[SERVER] Full_msg start: {full_msg[:32].hex()}")

        await send_length_prefixed(writer, full_msg)
        
        # Derive keys using transcript_hash (SAME on both sides now!)
        client_key = hkdf_sha256(shared_secret, b"", b"client->server" + transcript_hash, 32)
        server_key = hkdf_sha256(shared_secret, b"", b"server->client" + transcript_hash, 32)
        send_ctx = AEADContext(server_key)
        recv_ctx = AEADContext(client_key)
        self.aead_ctx = AEADPair(send_ctx, recv_ctx)
        self.session_key = transcript_hash
        self.state = TransferState.HANDSHAKE_COMPLETE
        self.writer_active = True
        self.handshake_done = True 
        logger.info(f"[SERVER] AEAD keys derived, session_key: {self.session_key.hex()[:16]}")
        logger.info(f"[SERVER] AEAD send_seq={self.aead_ctx.send_seq}, recv_seq={self.aead_ctx.recv_seq}")
        logger.info(f"[SERVER] State updated to {self.state.name}")

        ack = await self.recv_protected(reader)
        if ack != b"HANDSHAKE_OK": self._error("No ack")
        logger.info(f"[SERVER] Received handshake ACK: {ack}")

    async def _client_recv_response(self, reader, writer, **kwargs):
        data = await recv_length_prefixed(reader)
        logger.info(f"[CLIENT] Received handshake response, {len(data)} bytes")
        resp_msg = data[:-ML_DSA_65_SIG_LEN]
        signature = data[-ML_DSA_65_SIG_LEN:]
        #logger.info(f"[CLIENT] Server signature received, start: {signature.hex()[:16]}")
        #logger.info(f"[CLIENT] Response message length: {len(resp_msg)} bytes")

        ciphertext, server_dsa_pk = parse_handshake_resp(resp_msg)
        #logger.info(f"[CLIENT] Parsed server handshake resp, ciphertext start: {ciphertext.hex()[:16]}")
        #logger.info(f"[CLIENT] Server DSA key (parsed): {server_dsa_pk.hex()[:16]}")

        verify_peer_key(server_dsa_pk, self.key_path, "server", ephemeral=True)
        #logger.info(f"[CLIENT] TOFU verification passed for server key")

        domain = b"server handshake response"
        if not verify_message(hashlib.sha256(domain + resp_msg).digest(), signature, server_dsa_pk):
            self._error("Server sig invalid")
        #logger.info(f"[CLIENT] Server signature verified successfully")
        #print(f"[CLIENT] transcript before keys: {self.transcript.hexdigest()}")
        # CRITICAL: Hash resp_msg ONLY (matches server)
        self.transcript.update(resp_msg)  # ← Already correct!
        #print(f"[CLIENT] transcript before keys: {self.transcript.hexdigest()}")
        #logger.info(f"[CLIENT] Transcript updated, digest: {self.transcript.hexdigest()}")

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
        self.writer_active = True
        #logger.info(f"[CLIENT] Derived AEAD keys, session_key: {self.session_key.hex()[:16]}")
        #logger.info(f"[CLIENT] AEAD send_seq={self.aead_ctx.send_seq}, recv_seq={self.aead_ctx.recv_seq}")
        #logger.info(f"[CLIENT] State updated to {self.state.name}")

        await self.send_protected(reader, writer, b"HANDSHAKE_OK")
        logger.info(f"{self.role.upper()}: Handshake complete")

    def _error(self, reason: str):
        self.state = TransferState.ERROR
        raise ProtocolError(reason)

    def is_ready_for_transfer(self) -> bool:
        # Only allow sending if handshake complete AND writer is active
        return self.state == TransferState.HANDSHAKE_COMPLETE and self.aead_ctx is not None and self.writer_active
        
    def get_aead_context(self) -> AEADPair:
        if not self.aead_ctx:
            self._error("AEAD context not initialized")
        return self.aead_ctx
    
    async def send_protected(self, reader, writer, payload: bytes):
        """ALL post-handshake sends go through this"""
        if not self.is_ready_for_protected():
            raise ProtocolError("Handshake required first")
        seq = self.aead_ctx.send_seq.to_bytes(8, 'big')
        self.aead_ctx.send_seq += 1
        nonce = hkdf_sha256(seq, self.session_key , b"nonce", 12)
        encrypted = self.aead_ctx.send_ctx.encrypt(payload, nonce)
        msg = seq + nonce + encrypted 
        await send_length_prefixed(writer, msg)

    async def recv_protected(self, reader, writer=None) -> bytes:
        """ALL post-handshake receives go through this"""
        if self.state != TransferState.HANDSHAKE_COMPLETE or not self.aead_ctx:
            raise ProtocolError("Handshake required")
        msg = await recv_length_prefixed(reader)
        if msg is None:
            raise ProtocolError("Connection closed by peer")
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
        """Send file post-handshake using chunked AEAD encryption"""
        if not self.is_ready_for_transfer():
            raise ProtocolError("Handshake required before file transfer")
        
        await chunked_send_file(
            writer=writer,
            filepath=Path(filepath),
            aead_ctx=self.aead_ctx.send_ctx
        )

    async def _recv_file(self, reader, writer, output_path: str):
        """Receive file post-handshake using chunked AEAD decryption"""
        if not self.is_ready_for_transfer():
            raise ProtocolError("Handshake required before file transfer")
        await recv_file(
            reader=reader,        # USE THE ARGUMENT
            writer=writer,        # USE THE ARGUMENT
            output_path=Path(output_path),
            aead_ctx=self.aead_ctx.recv_ctx
        )
        # For AEAD sends (broadcast_weights, send_data)
    def is_ready_for_protected(self):
        return self.handshake_done and self.aead_ctx is not None

    # For file transfers
    def is_ready_for_file_transfer(self):
        return self.handshake_done and self.aead_ctx is not None and self.writer_active


