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
        self.current_round = 0  # Initialize round counter

    # -----------------------------
    # Persistent FL Connection
    # -----------------------------
    async def connect_fl(self, host: str, port: int = DEFAULT_PORT, initial_model_path: str = "model.npz"):
        self.mgr = SessionManager("client", self.key_path)
        
        # 1. Handshake + READY
        await self.mgr.establish_channel(host=host, port=port)
        logger.info("Handshake complete with server")
        await self.mgr.send_data(b'READY')
        logger.info("WORKER: Sent READY, waiting for master commands...")
        
        # 2. PERSISTENT event loop - NEVER exit
        while True:
            try:
                logger.info("WORKER: Waiting for data from master...")
                data = await self.mgr.recv_data()
                logger.info(f"WORKER: Received {len(data)} bytes of data")
                
                # Try to parse as JSON for message type
                try:
                    msg_obj = json.loads(data.decode())
                    msg_type = msg_obj.get("type")
                    payload_hex = msg_obj.get("payload_hex")
                    payload_text = msg_obj.get("payload", "")
                    logger.info(f"WORKER: Parsed JSON message type='{msg_type}'")
                    
                    if msg_type == "MODEL_FILE":
                        # Model file sent as hex in JSON
                        if payload_hex:
                            model_bytes = bytes.fromhex(payload_hex)
                        elif payload_text:
                            model_bytes = payload_text.encode()
                        else:
                            logger.error("WORKER: MODEL_FILE received but no payload found")
                            continue
                        
                        logger.info(f"WORKER: Received MODEL_FILE ({len(model_bytes)} bytes)")
                        model_path = Path(initial_model_path)
                        with open(model_path, "wb") as f:
                            f.write(model_bytes)
                        logger.info(f"WORKER: Saved model {model_path}")
                        
                        # Train immediately when model received
                        updated_bytes = await self.weights_callback(model_bytes, self.current_round)
                        await self._send_update_json(updated_bytes, round_no=self.current_round)
                        self.current_round += 1 
                        
                    elif msg_type == "AGGREGATED_MODEL":
                        # Model file sent as hex in JSON
                        if payload_hex:
                            model_bytes = bytes.fromhex(payload_hex)
                        elif payload_text:
                            model_bytes = payload_text.encode()
                        else:
                            logger.error("WORKER: AGGREGATED_MODEL received but no payload found")
                            continue
                        
                        logger.info(f"WORKER: Received AGGREGATED_MODEL ({len(model_bytes)} bytes)")
                        model_path = Path(initial_model_path)
                        with open(model_path, "wb") as f:
                            f.write(model_bytes)
                        logger.info("WORKER: New global model saved, ready for next round")
                    else:
                        logger.debug(f"WORKER: Ignoring unknown message type '{msg_type}'")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    # Not JSON - might be raw binary data (shouldn't happen with this protocol)
                    logger.debug(f"WORKER: Received {len(data)} bytes of raw data (not JSON): {e}")
                    
            except Exception as e:
                logger.error(f"WORKER loop error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Retry



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
    # this is where the transferlearning model will be defined NOT TRAINED AND THEN ITS CALLED TO
    # THE RUN_WORKER FUNCTION TO BE TRAINED NOW OK??? AND 
    # THEN THE CALLBACK WILL BE CALLED WHEN WEIGHTS ARE RECEIVED
    # THEN IN 
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

