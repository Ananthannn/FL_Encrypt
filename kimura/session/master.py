import asyncio
import logging
from pathlib import Path
from kimura.session.manager import SessionManager
from kimura.protocol.constants import DEFAULT_PORT
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SecureServer:
    def __init__(self, key_path: str, base_output: str = None):
        self.key_path = Path(key_path)
        self.base_output = Path(base_output) if base_output else None
        self.clients_processed = 0
        self.active_clients = {}  
        self.on_weights_received = None  
        
    async def send_to_client(self, client_id: int, data: bytes):
        if client_id not in self.active_clients:
            return
        reader, writer, sm = self.active_clients[client_id]
        # Use your existing state_machine send_signed_data (add later)
        await sm.send_signed_data(writer, data)
    
    async def broadcast_weights(self, weights: bytes):
        for client_id in list(self.active_clients.keys()):
            await self.send_to_client(client_id, weights)
    
    # master.py - FIXED handle_client()
    async def handle_client(self, reader, writer):
        client_id = self.clients_processed
        self.clients_processed += 1
        logger.info(f"Client #{client_id} connected")
        
        # ✅ THIS WAS MISSING - create SessionManager per client
        mgr = SessionManager("server", str(self.key_path), self.base_output)
        
        try:
            # ✅ THIS triggers server handshake (recv_handshake → send_response)
            await mgr.establish_channel(reader=reader, writer=writer)
            logger.info(f"Client #{client_id}: Handshake COMPLETED ✅")
            
            # ✅ Now receive file
            output_file = self.base_output.parent / f"{self.base_output.stem}_worker{client_id}.pt"
            await mgr.recv_file(output_file)
            
        except asyncio.IncompleteReadError:
            logger.error(f"Client #{client_id} disconnected: Connection closed by client")
        except Exception as e:
            logger.error(f"Client #{client_id} disconnected: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve_forever(self, port: int = DEFAULT_PORT, host: str = "0.0.0.0"):
        server = await asyncio.start_server(self.handle_client, host, port)
        logger.info(f"Server listening on {host}:{port}")

        async with server:
            await server.serve_forever()

    async def send_to_client(self, client_id: int, data: bytes):
        if client_id not in self.active_clients:
            logger.warning(f"Client #{client_id} not active")
            return
        reader, writer, sm = self.active_clients[client_id]
        await sm.send_signed_data(writer, data)
        logger.info(f"Sent {len(data)/1024/1024:.1f}MB to Client #{client_id}")  # ADD

    async def broadcast_weights(self, weights: bytes):
        sent_count = 0
        for client_id in list(self.active_clients.keys()):
            await self.send_to_client(client_id, weights)
            sent_count += 1
        logger.info(f"Broadcast {len(weights)/1024/1024:.1f}MB to {sent_count} clients")
