#!/usr/bin/env python3
import asyncio
import json
import logging
from pathlib import Path
from kimura.session.manager import SessionManager
from kimura.protocol.constants import DEFAULT_PORT
import warnings
from kimura.protocol.fl_protocol import (
    FLMessageType,
    serialize_fl_message,
    parse_fl_message
)
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SecureServer:
    """
    FL Master Server

    Responsibilities:
    - Send initial full model to all clients
    - Receive gradients/updates from clients
    - Broadcast updated weights/deltas
    - Maintain per-client state
    """
    def __init__(self, key_path: str, base_output: str = None):
        self.key_path = Path(key_path)
        self.base_output = Path(base_output) if base_output else None

        self.active_clients = {}  # worker_id -> (reader, writer, SessionManager)
        self.client_states = {}   # worker_id -> WorkerState

        # Callbacks
        self.on_worker_connected = None
        self.on_worker_ready = None
        self.on_result_received = None
        self.on_weights_received = None

    # ===============================
    # CLIENT CONNECTION HANDLING
    # ===============================
    async def handle_client(self, reader, writer):
        mgr = SessionManager("server", str(self.key_path), self.base_output)
        
        try:
            await mgr.establish_channel(reader=reader, writer=writer)
            # Extract worker_id from SessionManager (derived from peer pubkey during handshake)
            worker_id = mgr.worker_id
            self.active_clients[worker_id] = (reader, writer, mgr)
            
            logger.info(f"Worker {worker_id} handshake complete")
            if self.on_worker_connected:
                await self.on_worker_connected(worker_id)
                  
            # LOOP for multiple rounds - DON'T EXIT HERE
            while worker_id in self.active_clients:  # Keep alive
                try:
                    # Use timeout to allow server to send files between receives
                    data = await asyncio.wait_for(mgr.recv_data(), timeout=60.0)
                    # Check if worker is signaling READY for initial task
                    try:
                        if not data:
                            continue
                        msg_type, payload = parse_fl_message(data)
                    except Exception as e:
                        logger.error(f"Invalid FL message from {worker_id}: {e}")
                        continue

                    if msg_type == FLMessageType.MODEL_LOADED:
                        if self.on_worker_ready:
                            await self.on_worker_ready(worker_id, payload)

                    elif msg_type == FLMessageType.UPDATE:
                        if self.on_result_received:
                            await self.on_result_received(worker_id, payload)

                    else:
                        logger.warning(f"Unhandled message {msg_type} from {worker_id}")

                except asyncio.TimeoutError:
                    # Worker is idle, but connection is still alive - continue listening
                    await asyncio.sleep(0.1)
                    continue
                except asyncio.IncompleteReadError:
                    logger.info(f"Worker {worker_id} completed rounds normally (EOF received)")
                    break  # Worker finished cleanly - exit loop
                except Exception as e:
                    logger.error(f"Worker {worker_id} recv error: {e}")
                    break  # Other network/protocol error
        
        except Exception as e:
            logger.error(f"Worker handshake/connection error: {e}")
        finally:
            # Only close if worker was in active list
            for wid in list(self.active_clients.keys()):
                r, w, m = self.active_clients[wid]
                if w is writer:  # Find and remove by writer
                    del self.active_clients[wid]
                    logger.info(f"Worker {wid} disconnected")
                    break
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

    async def send_to_worker(self, worker_id: str, msg_type: FLMessageType, payload: bytes):
        if worker_id not in self.active_clients:
            logger.warning(f"Worker {worker_id} not active")
            return
        _, writer, mgr = self.active_clients[worker_id]
        try:
            fl_bytes = serialize_fl_message(msg_type, payload)
            await mgr.send_data(fl_bytes)
            logger.info(f"Sent {msg_type.name} to Worker {worker_id}")
        except Exception as e:
            logger.error(f"Failed to send {msg_type.name} to {worker_id}: {e}")

    # ===============================
    # SERVER LOOP
    # ===============================
    async def serve_forever(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0"):
        server = await asyncio.start_server(self.handle_client, host, port)
        logger.info(f"Server listening on {host}:{port}")
        async with server:
            await server.serve_forever()

    # ===============================
    # SEND / RECEIVE UTILITIES
    # ===============================
    async def send_file(self, worker_id: str, file_path: str):
        """Send large model (initial round) to a specific worker"""
        if worker_id not in self.active_clients:
            logger.warning(f"Worker {worker_id} not active")
            return
        _, writer, mgr = self.active_clients[worker_id]
        await mgr.send_file(file_path)
        logger.info(f"Sent full model ({Path(file_path).name}) to Worker {worker_id}")
    
    async def broadcast_weights(self, weights: bytes):
        sent_count = 0
        dead_clients = []
        
        for worker_id in list(self.active_clients.keys()):
            if worker_id not in self.active_clients:
                continue
                
            _, writer, mgr = self.active_clients[worker_id]
            try:
                # Check if connection alive + handshake complete
                if mgr.state_machine.is_ready_for_protected():
                    fl_bytes = serialize_fl_message(
                        FLMessageType.AGGREGATED_MODEL,
                        weights
                    )
                    await mgr.send_data(fl_bytes)
                    sent_count += 1
                else:
                    logger.warning(f"Worker {worker_id} not ready")
                    dead_clients.append(worker_id)
            except Exception as e:
                logger.error(f"Worker {worker_id} broadcast failed: {e}")
                dead_clients.append(worker_id)
        
        # Clean dead clients
        for wid in dead_clients:
            if wid in self.active_clients:
                _, writer, _ = self.active_clients[wid]
                del self.active_clients[wid]
                writer.close()
        
        logger.info(f"Sent {len(weights)/1024:.1f}KB to {sent_count} workers")

