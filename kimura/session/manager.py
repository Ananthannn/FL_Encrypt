# session/manager.py - COMPLETE flow w/ your tcp.py
import asyncio
import logging
from pathlib import Path
from kimura.file_transfer.transfer import chunked_send_file, recv_file
from kimura.transport.tcp import TCPTransport
from kimura.protocol.state_machine import StateMachine
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
        if self.role == "client":
            # connect normally
            self.reader, self.writer = await self.transport.connect(host or "127.0.0.1", port)

            # SEND handshake & internally process server response
            await self.state_machine.transition("send_handshake", reader=self.reader, writer=self.writer)

            # NO need to call recv_response manually; StateMachine already does it
            logger.info(f"{self.role.upper()}: Handshake completed")

        else:
            # SERVER: pre-connected streams
            if not (reader and writer):
                raise ValueError("Server: must provide reader/writer from handle_client")
            self.reader, self.writer = reader, writer

            await self.state_machine.transition("recv_handshake", reader=self.reader, writer=self.writer)
            await self.state_machine.transition("send_response", reader=self.reader, writer=self.writer)
            logger.info(f"{self.role.upper()}: Handshake completed")
            self.writer.write(b"READY")
            await self.writer.drain()
            logger.info(f"{self.role.upper()}: Sent READY signal to client")
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
        """Client sends file post-handshake using chunked AEAD encryption."""
        if not self.state_machine.is_ready_for_transfer():
            raise RuntimeError("Handshake required first")
        # Use the AEAD send context from the state machine (already created during handshake)
        await self.state_machine.transition(
            "start_send_file",
            reader=self.reader,
            writer=self.writer,
            filepath=str(filepath)
        )
        logger.info(f"CLIENT sent {Path(filepath).name}")

    async def recv_file(self, output_path: str):
        """Server receives file post-handshake using StateMachine wrapper."""
        if not self.state_machine.is_ready_for_transfer():
            raise RuntimeError("Handshake required first")

        await self.state_machine.transition(
            "start_recv_file",
            reader=self.reader,
            writer=self.writer,
            output_path=str(output_path)
        )
        logger.info(f"SERVER received file -> {output_path}")

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