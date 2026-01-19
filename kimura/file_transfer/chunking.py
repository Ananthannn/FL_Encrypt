# file_transfer/chunking.py - FIXED
"""
Production-grade chunking for PQC secure file transfer.
Each chunk independently encryptable with unique nonce.
"""

import os
import hashlib
from typing import Iterator, List, Tuple, NamedTuple, Union
from pathlib import Path
from protocol.constants import MAX_CHUNK_SIZE
# Forward reference for your AEADContext (no import needed)
from typing import ForwardRef
AEADContext = ForwardRef("AEADContext")


class ChunkMetadata(NamedTuple):
    """Chunk metadata for reassembly and integrity."""
    index: int           # Sequence position (0, 1, 2...)
    size: int            # Original chunk size in bytes
    hash: bytes          # SHA-256 of original chunk


def chunk_bytes(
    data: bytes,
    chunk_size: int = MAX_CHUNK_SIZE,  # 64KB optimal for AES-GCM
) -> List[Tuple[int, bytes]]:
    """Split bytes into independently encryptable chunks."""
    return [(i // chunk_size, data[i:i + chunk_size]) 
            for i in range(0, len(data), chunk_size)]


def chunk_bytes_with_metadata(
    data: bytes,
    chunk_size: int = MAX_CHUNK_SIZE,
) -> List[Tuple[ChunkMetadata, bytes]]:
    """Chunk with full metadata for production transfers."""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        metadata = ChunkMetadata(
            index=i // chunk_size,
            size=len(chunk),
            hash=hashlib.sha256(chunk).digest()
        )
        chunks.append((metadata, chunk))
    return chunks


def reassemble_chunks(
    indexed_chunks: List[Tuple[int, bytes]],
    verify_order: bool = True,
) -> bytes:
    """Reassemble chunks in correct order."""
    if verify_order:
        indices = sorted(idx for idx, _ in indexed_chunks)
        if indices != list(range(len(indices))):
            raise ValueError("Chunk indices incomplete or out of order")
    return b''.join(chunk for _, chunk in sorted(indexed_chunks))


def stream_chunks(
    filepath: Union[str, Path],
    chunk_size: int = MAX_CHUNK_SIZE,
) -> Iterator[Tuple[int, bytes]]:
    """Stream chunks directly from file (zero-copy)."""
    with open(filepath, 'rb') as f:
        for i, chunk in enumerate(iter(lambda: f.read(chunk_size), b'')):
            yield i, chunk


def verify_chunk_integrity(
    metadata: ChunkMetadata,
    chunk: bytes,
) -> bool:
    """Verify single chunk integrity."""
    return metadata.hash == hashlib.sha256(chunk).digest()


def chunk_file_for_encryption(
    filepath: Union[str, Path],
    chunk_size: int = MAX_CHUNK_SIZE,
) -> Iterator[Tuple[ChunkMetadata, bytes]]:
    """
    Production pipeline: file → chunks → AES-GCM ready.
    """
    with open(filepath, 'rb') as f:
        for i, chunk in enumerate(iter(lambda: f.read(chunk_size), b'')):
            yield ChunkMetadata(i, len(chunk), hashlib.sha256(chunk).digest()), chunk


# FIXED: Standalone - no AEADContext dependency
def reassemble_chunks_with_hashes(
    chunks_with_metadata: List[Tuple[ChunkMetadata, bytes]],
    verify_integrity: bool = True,
) -> bytes:
    """
    Reassemble + verify (no crypto dependency).
    
    Args:
        chunks_with_metadata: List of (metadata, plaintext_chunk)
        verify_integrity: Check SHA-256 per chunk
    """
    if verify_integrity:
        for metadata, chunk in chunks_with_metadata:
            if not verify_chunk_integrity(metadata, chunk):
                raise ValueError(f"Integrity fail at chunk {metadata.index}")
    
    # Sort by index and join
    sorted_chunks = sorted(chunks_with_metadata, key=lambda x: x[0].index)
    return b''.join(chunk for _, chunk in sorted_chunks)


# Your ACTUAL PQC Pipeline (separate integration functions)
def encrypt_pipeline(filepath: str, aead_ctx, chunk_size: int = MAX_CHUNK_SIZE):
    """Complete sender pipeline."""
    encrypted_chunks = []
    for metadata, chunk in chunk_file_for_encryption(filepath, chunk_size):
        nonce = aead_ctx.generate_nonce()
        encrypted = aead_ctx.encrypt(chunk, nonce)
        encrypted_chunks.append((metadata, nonce, encrypted))
    return encrypted_chunks


def decrypt_pipeline(
    encrypted_chunks: List[Tuple[ChunkMetadata, bytes, bytes]], 
    aead_ctx
) -> bytes:
    """Complete receiver pipeline."""
    chunks = []
    for metadata, nonce, encrypted in encrypted_chunks:
        chunk = aead_ctx.decrypt(encrypted, nonce)
        if not verify_chunk_integrity(metadata, chunk):
            raise ValueError(f"Integrity fail at chunk {metadata.index}")
        chunks.append((metadata, chunk))
    return reassemble_chunks_with_hashes(chunks)

    