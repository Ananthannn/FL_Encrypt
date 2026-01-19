# transport/tcp.py
"""asyncio TCP transport layer for PQC file transfer."""
import socket
import asyncio
from typing import Tuple, Optional
import logging
logger = logging.getLogger(__name__)
class TCPTransport:
    """High-level TCP transport abstraction."""
    @staticmethod
    async def connect(host: str, port: int, timeout: float = 30.0) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect to TCP server w/ ADVANCED PERFORMANCE TUNING."""
        try:
            loop = asyncio.get_running_loop()
            
            # METHOD 1: Pre-created socket (BEST performance)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setblocking(False)
                
                # ULTRA TCP TUNING:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)                    # Disable Nagle
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)                   # Immediate ACKs  
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8*1024*1024)             # 8MB send
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8*1024*1024)             # 8MB recv
                sock.setsockopt(socket.SOL_TCP, socket.TCP_MAXSEG, 1440)                      # Optimal MTU
                
                # Connect THEN tune (avoids sock= parameter conflict)
                await loop.sock_connect(sock, (host, port))
                reader, writer = await asyncio.wait_for(
                    loop.create_connected_stream_transport(sock, None),
                    timeout=timeout
                )
                
            except (AttributeError, ValueError, NotImplementedError):
                # METHOD 2: Post-connection tuning (fallback)
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout
                )
                sock = writer.transport.get_extra_info('socket')
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8*1024*1024)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8*1024*1024)
            
            addr = writer.get_extra_info('peername')
            logger.info(f"Connected to {addr} (optimized)")
            return reader, writer
            
        except asyncio.TimeoutError:
            raise ConnectionError(f"Timeout connecting to {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Connection failed: {e}")


    @staticmethod
    async def serve(host: str, port: int, handler: callable, backlog: int = 128) -> asyncio.AbstractServer:
        """Start TCP server."""
        server = await asyncio.start_server(handler, host, port, backlog=backlog)
        addr = server.sockets[0].getsockname()
        logger.info(f"TCP Server listening on {addr}")
        return server
    
    @staticmethod
    async def safe_close(writer: asyncio.StreamWriter) -> None:
        """Safely close writer."""
        if writer.is_closing():
            return
        writer.close()
        try:
            await writer.wait_closed()
        except asyncio.CancelledError:
            pass

# Connection context manager
class TCPConnection:
    """Context manager for TCP connections."""
    
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
    
    async def __aenter__(self) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.reader, self.writer = await TCPTransport.connect(self.host, self.port, self.timeout)
        return self.reader, self.writer
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            await TCPTransport.safe_close(self.writer)

# Server context manager  
class TCPServer:
    """Context manager for TCP servers."""
    
    def __init__(self, host: str, port: int, handler: callable):
        self.host = host
        self.port = port
        self.handler = handler
        self.server: Optional[asyncio.AbstractServer] = None
    
    async def __aenter__(self) -> asyncio.AbstractServer:
        self.server = await TCPTransport.serve(self.host, self.port, self.handler)
        return self.server
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.server:
            self.server.close()
            await self.server.wait_closed()

