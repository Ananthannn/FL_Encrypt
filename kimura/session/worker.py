import asyncio
from .manager import SessionManager
from protocol.constants import DEFAULT_PORT
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
import logging
logger = logging.getLogger(__name__)
class Worker:
    def __init__(self, key_path: str, file_path: str = None):  # Make file_path optional
        self.key_path = key_path
        self.file_path = file_path
        self.mgr = None
        self.on_weights_received = None  # FL callback
        
    async def connect_and_send(self, host, port):
        self.mgr = SessionManager("client", self.key_path)

        # 1) Do handshake ONCE
        await self.mgr.establish_channel(host=host, port=port)

        # 2) WAIT for server readiness (CRUCIAL)
        ready = await self.mgr.reader.readexactly(5)
        if ready != b"READY":
            raise RuntimeError(f"Server not ready (got: {ready})")
        logger.info("Server ready, starting file transfer...")
        # 3) NOW send file
        if self.file_path:
            await self.mgr.send_file(str(self.file_path))
            logger.info(f"File {self.file_path} sent successfully")

        # 4) Graceful close
        await self.mgr.close()
        logger.info("Connection closed after file transfer")

    # FL PERSISTENT MODE (doesn't close connection)
    async def connect_fl(self, host: str, port: int = DEFAULT_PORT):
        """FL mode: Persistent bidirectional channel"""
        self.mgr = SessionManager("client", self.key_path)
        await self.mgr.establish_channel(host=host, port=port)
        
        # Start FL loop in background
        asyncio.create_task(self._fl_loop())
    
    async def send_weights(self, weights: bytes):
        """NEW: Send weights over existing channel"""
        if self.mgr:
            # You'll add this to SessionManager later
            await self.mgr.send_data(weights)
    
    def set_weights_callback(self, callback):
        """Callback when server sends weights"""
        self.on_weights_received = callback
    
    async def _fl_loop(self):
        """Internal FL bidirectional loop"""
        while True:
            try:
                # Receive from server
                weights = await self.mgr.recv_data()  # You'll add this
                if self.on_weights_received:
                    trained = await self.on_weights_received(weights)
                    await self.send_weights(trained)
            except:
                break
