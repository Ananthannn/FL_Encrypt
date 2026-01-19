# session/manager.py - COMPLETE flow w/ your tcp.py
import asyncio
import logging
from pathlib import Path
from file_transfer.transfer import send_length_prefixed
from protocol.messages import serialize_handshake_init
from transport.tcp import TCPTransport
from protocol.state_machine import StateMachine
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
from protocol.constants import DEFAULT_PORT
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)-8s %(name)s %(message)s',  # ADD %(name)s
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, role: str, key_path: str = "./keys", output_path: str = None):
        self.role = role
        self.key_path = key_path
        self.output_path = output_path
        self.ready = asyncio.Event()
        self.state_machine = StateMachine(key_path, role)
        self.transport = TCPTransport()
        self.server_running = False
        self.active_clients = {}  # {client_id: (reader, writer, state_machine)}
        self.client_counter = 0
    
    async def establish_channel(self, reader=None, writer=None, host=None, port=DEFAULT_PORT):
        """Accept pre-connected streams OR connect as client"""
        if self.role == "client":
            # Client still connects normally
            self.reader, self.writer = await self.transport.connect(host or "127.0.0.1", port)
            await self.state_machine.transition("send_handshake", reader=self.reader, writer=self.writer)
            await self.state_machine.transition("recv_response", reader=self.reader, writer=self.writer)
            logger.info(f"{self.role.upper()}: Handshake completed")
        else:
            # SERVER - REQUIRE pre-connected reader/writer from handle_client
            if not (reader and writer):
                raise ValueError("Server: must provide reader/writer from handle_client")
            self.reader, self.writer = reader, writer
            await self.state_machine.transition("recv_handshake", reader=self.reader, writer=self.writer)
            await self.state_machine.transition("send_response", reader=self.reader, writer=self.writer)
            logger.info(f"{self.role.upper()}: Handshake completed")
        self.ready.set()

    
    async def _client_handshake(self):
        """CLIENT: Send handshake via StateMachine (handles signing automatically)."""
        self.reader, self.writer = await self.transport.connect("127.0.0.1", DEFAULT_PORT)
        await self.state_machine.transition("send_handshake", 
                                        reader=self.reader, 
                                        writer=self.writer)
        
        await self.state_machine.transition("recv_response", 
                                        reader=self.reader, 
                                        writer=self.writer)
        logger.info(f"{self.role.upper()}: Handshake completed")

    async def send_file(self, filepath: str):
        """Client sends file post-handshake"""
        if not self.state_machine.is_ready_for_transfer():
            raise RuntimeError("Handshake required first")
        
        with open(filepath, 'rb') as f:
            file_data = f.read()
        
        filename = Path(filepath).name.encode()
        filename_len = len(filename)
        data_len = len(file_data)
        
        # Format: filename_len(4) + filename + data_len(8) + data
        msg = (
            filename_len.to_bytes(4, 'big') + 
            filename + 
            data_len.to_bytes(8, 'big') + 
            file_data
        )
        
        await self.state_machine.send_protected(self.reader, self.writer, msg)
        logger.info(f"CLIENT sent {Path(filepath).name} ({len(file_data)/1024/1024:.1f}MB)")


    async def recv_file(self, output_path: str):
        """Server receives file post-handshake using THIS SessionManager's state_machine"""
        if not self.state_machine.is_ready_for_transfer():
            raise RuntimeError("Handshake required first")
        
        # Receive file using THIS session's reader (not active_clients!)
        msg = await self.state_machine.recv_protected(self.reader)
        
        # Parse: filename_len(4) + filename + data_len(8) + data
        filename_len = int.from_bytes(msg[:4], 'big')
        filename = msg[4:4+filename_len].decode()
        data_offset = 4 + filename_len
        data_len = int.from_bytes(msg[data_offset:data_offset+8], 'big')
        file_data = msg[data_offset+8:data_offset+8+data_len]
        
        with open(output_path, 'wb') as f:
            f.write(file_data)
        logger.info(f"SERVER received {filename} ({len(file_data)/1024/1024:.1f}MB) -> {output_path}")


    async def close(self):
        logger.info("SessionManager shutting down")
        for client_id, (reader, writer, sm) in list(self.active_clients.items()):
            try:
                if writer:
                    writer.close()
                    await writer.wait_closed()
                    logger.info(f"Closed client #{client_id}")
            except Exception as e:
                logger.warning(f"Error closing client #{client_id}: {e}")
        self.active_clients.clear()
        try:
            if hasattr(self, "writer") and self.writer:
                self.writer.close()
                await self.writer.wait_closed()
                self.writer = None
                logger.info("Closed client writer")
        except Exception as e:
            logger.warning(f"Error closing main writer: {e}")
        # 4. Reset readiness
        if self.ready:
            self.ready.clear()
        logger.info("SessionManager cleanup complete")

    async def send_data(self, data: bytes):
        """Send signed weights"""
        await self.state_machine.send_signed_data(self.writer, data)
        logger.info(f"{self.role.upper()}: Sent {len(data)/1024/1024:.1f}MB")

    async def recv_data(self) -> bytes:
        """Receive + verify weights"""
        data = await self.state_machine.recv_and_verify_data(self.reader)
        logger.info(f"{self.role.upper()}: Received {len(data)/1024/1024:.1f}MB")
        return data