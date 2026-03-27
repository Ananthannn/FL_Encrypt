# file_transfer/transfer.py
"""
secure file transfer over asyncio TCP.
"""

import asyncio
import struct
import os
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
# Fix imports for project structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))  # file_transfer/
from kimura.crypto.aead import AEADContext
from kimura.file_transfer.chunking import chunk_file_for_encryption, ChunkMetadata, verify_chunk_integrity
from kimura.file_transfer.bytes_conversion import format_file_size, get_file_size


async def send_length_prefixed(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Send data with 4-byte length prefix."""
    length = len(data)
    writer.write(struct.pack('>I', length))  # 4-byte length
    writer.write(data)
    await writer.drain()
    
async def recv_length_prefixed(reader: asyncio.StreamReader) -> bytes | None:
    try:
        length_data = await reader.readexactly(4)
    except asyncio.IncompleteReadError as e:
        if len(e.partial) == 0:
            logger.info("recv_length_prefixed: peer closed connection (EOF)")
            return None
        raise

    length = struct.unpack('>I', length_data)[0]

    if length == 0:
        logger.warning("recv_length_prefixed: zero-length frame")
        return b""

    logger.debug(f"recv_length_prefixed: expecting {length} bytes")

    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        logger.warning("recv_length_prefixed: truncated frame")
        return None

async def chunked_send_file(
    writer: asyncio.StreamWriter,
    filepath: Path,
    aead_ctx: AEADContext,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Send encrypted file in chunks, log only at the end."""
    file_size = get_file_size(filepath)
    total_chunks = (file_size + chunk_size - 1) // chunk_size

    # 1. Send file header: total_chunks, file_size
    header = struct.pack('>QQ', total_chunks, file_size)
    await send_length_prefixed(writer, header)

    # 2. Send encrypted chunks
    chunks_sent = 0
    for metadata, chunk in chunk_file_for_encryption(filepath, chunk_size):
        nonce = aead_ctx.generate_nonce()
        encrypted = aead_ctx.encrypt(chunk, nonce)
        msg = (
            metadata.index.to_bytes(8, 'big') +
            metadata.size.to_bytes(8, 'big') +
            metadata.hash +
            nonce +
            encrypted
        )
        await send_length_prefixed(writer, msg)
        chunks_sent += 1

    # 3. Log once at the end
    logger.info(
        f"File sent: {filepath.name}, "
        f"{chunks_sent}/{total_chunks} chunks, "
        f"size={format_file_size(file_size)}"
    )


async def recv_file(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    output_path: Path,
    aead_ctx: AEADContext,
) -> Path:
    """✅ FULL integrity verification with ChunkMetadata."""
    # 1. Read header
    header = await recv_length_prefixed(reader)
    if header is None:
        logger.info("recv_file: EOF - no more data expected")
        return  # MASTER FINISHED SENDING
    total_chunks, expected_size = struct.unpack('>QQ', header)
    
    chunks = []
    for i in range(total_chunks):
        # 2. Read encrypted chunk packet
        msg = await recv_length_prefixed(reader)
        # Parse packet format
        index = int.from_bytes(msg[:8], 'big')
        size = int.from_bytes(msg[8:16], 'big')
        chunk_hash = msg[16:48]                           # SHA256 hash
        nonce = msg[48:60]                                # 12-byte nonce
        encrypted = msg[60:]                              # Encrypted data
        # 3. Decrypt
        chunk = aead_ctx.decrypt(encrypted, nonce)
        metadata = ChunkMetadata(index, size, chunk_hash)
        if not verify_chunk_integrity(metadata, chunk):
            raise ValueError(f"❌ Integrity fail at chunk {index}")
        chunks.append((index, chunk))
    # 5. Reassemble in order
    chunks.sort(key=lambda x: x[0])
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        for _, chunk in chunks:
            f.write(chunk)
    return output_path