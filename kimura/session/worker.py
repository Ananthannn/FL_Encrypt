import asyncio
from .manager import SessionManager
from protocol.constants import DEFAULT_PORT
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")

class PQCWorker:
    def __init__(self, key_path: str, file_path: str = None):  # Make file_path optional
        self.key_path = key_path
        self.file_path = file_path
        self.mgr = None
        self.on_weights_received = None  # FL callback
        
    async def connect_and_send(self, host: str, port: int = DEFAULT_PORT):
        """✅ KEEP EXISTING - file transfer mode"""
        self.mgr = SessionManager("client", self.key_path)
        try:
            await self.mgr.establish_channel(host=host, port=port)
            if self.file_path:
                await self.mgr.send_file(str(self.file_path))
        finally:
            await self.mgr.close()
    
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
