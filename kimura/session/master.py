#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
from kimura.session.manager import SessionManager
from kimura.protocol.constants import DEFAULT_PORT
from shared.state import WorkerState
import warnings

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

        self.clients_processed = 0
        self.active_clients = {}  # client_id -> (reader, writer, SessionManager)
        self.client_states = {}   # client_id -> WorkerState

        # Callbacks
        self.on_worker_connected = None
        self.on_worker_ready = None
        self.on_result_received = None
        self.on_weights_received = None

    # ===============================
    # CLIENT CONNECTION HANDLING
    # ===============================
    async def handle_client(self, reader, writer):
        client_id = self.clients_processed
        self.clients_processed += 1
        
        mgr = SessionManager("server", str(self.key_path), self.base_output)
        self.active_clients[client_id] = (reader, writer, mgr)
        
        try:
            await mgr.establish_channel(reader=reader, writer=writer)
            logger.info(f"Client #{client_id} handshake complete")
            if self.on_worker_connected:
                await self.on_worker_connected(str(client_id))
            # Send model
            model_path = self.base_output / "model.npz"
            await mgr.send_file(str(model_path))
            
            # LOOP for multiple rounds - DON'T EXIT HERE
            while client_id in self.active_clients:  # Keep alive
                try:
                    data = await mgr.recv_data()
                    if self.on_result_received:
                        await self.on_result_received(client_id, data)
                except asyncio.IncompleteReadError:
                    logger.info(f"Client #{client_id} completed rounds normally (EOF received)")
                    break  # Worker finished cleanly - exit loop
                except Exception as e:
                    logger.error(f"Client #{client_id} recv error: {e}")
                    break  # Other network/protocol error
        
        except Exception as e:
            logger.error(f"Client #{client_id} error: {e}")
        finally:
            # Only close if client was removed
            if client_id in self.active_clients:
                del self.active_clients[client_id]
                writer.close()
                await writer.wait_closed()

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
    async def send_file(self, client_id: int, file_path: str):
        """Send large model (initial round) to a specific client"""
        if client_id not in self.active_clients:
            logger.warning(f"Client #{client_id} not active")
            return
        _, writer, mgr = self.active_clients[client_id]
        await mgr.send_file(file_path)
        logger.info(f"Sent full model ({Path(file_path).name}) to Client #{client_id}")
    
    async def broadcast_weights(self, weights: bytes):
        sent_count = 0
        dead_clients = []
        
        for client_id in list(self.active_clients.keys()):
            if client_id not in self.active_clients:
                continue
                
            _, writer, mgr = self.active_clients[client_id]
            try:
                # Check if connection alive + handshake complete
                if mgr.state_machine.is_ready_for_protected():
                    await mgr.send_data(weights)
                    sent_count += 1
                else:
                    logger.warning(f"Client #{client_id} not ready")
                    dead_clients.append(client_id)
            except Exception as e:
                logger.error(f"Client #{client_id} broadcast failed: {e}")
                dead_clients.append(client_id)
        
        # Clean dead clients
        for cid in dead_clients:
            if cid in self.active_clients:
                _, writer, _ = self.active_clients[cid]
                del self.active_clients[cid]
                writer.close()
        
        logger.info(f"Sent {len(weights)/1024:.1f}KB to {sent_count} clients")

