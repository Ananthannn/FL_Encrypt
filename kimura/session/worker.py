#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
from kimura.session.manager import SessionManager
from kimura.protocol.constants import DEFAULT_PORT
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logger = logging.getLogger(__name__)


class SecureClient:
    def __init__(self, key_path: str):
        """
        FL Client: persistent bidirectional channel
        key_path: directory with PQC keys
        """
        self.key_path = key_path
        self.weights_callback: callable | None = None
        self.mgr: SessionManager | None = None
        self.on_weights_received = None  # callback for FL loop

    # -----------------------------
    # Persistent FL Connection
    # -----------------------------
    async def connect_fl(self, host: str, port: int = DEFAULT_PORT, initial_model_path: str = "model.npz"):
        self.mgr = SessionManager("client", self.key_path)

        # 1️⃣ Secure handshake
        await self.mgr.establish_channel(host=host, port=port)
        logger.info("Handshake complete with server")
        await self.mgr.send_data(b'READY')  # Notify server we're ready

        # 2️⃣ Receive initial model (FILE)
        logger.info("WORKER: waiting for initial task model")
        await self.mgr.recv_file(Path(initial_model_path))

        # 3️⃣ Train ONCE immediately
        if not self.weights_callback:
            raise RuntimeError("Weights callback not set!")
        logger.info("WORKER: training initial model")
        updated_bytes = await self.weights_callback(Path(initial_model_path).read_bytes())

        # 4️⃣ SEND FIRST UPDATE
        logger.info("WORKER: sending initial update to master")
        await self._send_update_json(updated_bytes, round_no=0)

        # 5️⃣ Now enter persistent loop
        await self._fl_loop()


    async def _send_update_json(self, weights: bytes, round_no: int):
        """
        Wrap weights in JSON with round_no and send.
        """
        if not self.mgr:
            raise RuntimeError("SessionManager not initialized")

        payload = {
            "round_no": round_no,
            "weights": weights.hex()
        }
        await self.mgr.send_data(json.dumps(payload).encode())
        logger.info(f"WORKER: Sent {len(weights)/1024/1024:.3f} MB for round {round_no}")
    
    # -----------------------------
    # Send updated gradients / weights
    # -----------------------------
    async def send_weights(self, weights: bytes):
        """
        Send local training updates back to the server.
        """
        if self.mgr:
            await self.mgr.send_data(weights)
            logger.info(f"Sent {len(weights)/1024:.1f} KB of gradients to server")

    # -----------------------------
    # Register callback for server updates
    # -----------------------------
    def set_weights_callback(self, callback: callable):
        """
        Set callback for handling received server weights.
        callback should be async and accept bytes -> returns bytes
        """
        self.weights_callback = callback

    # -----------------------------
    # Internal FL loop
    # -----------------------------
    async def _fl_loop(self):
        """
        Handles FL rounds AFTER round-0.
        Server always sends first here.
        """
        if not self.mgr:
            raise RuntimeError("FL connection not established")

        while True:
            try:
                logger.info("WORKER: waiting for aggregated weights")
                server_weights = await self.mgr.recv_data()

                if self.on_weights_received:
                    updated_weights = await self.on_weights_received(server_weights)

                    logger.info("WORKER: sending updated weights")
                    if not hasattr(self, "_current_round"):
                        self._current_round = 1  # round-0 already sent

                    await self._send_update_json(updated_weights, round_no=self._current_round)
                    self._current_round += 1

            except asyncio.IncompleteReadError:
                logger.warning("Server closed connection")
                break
            except Exception as e:
                logger.error(f"FL loop error: {e}")
                break

