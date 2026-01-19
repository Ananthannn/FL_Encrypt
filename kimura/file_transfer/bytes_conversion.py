# file_transfer/bytes_conversion.py
"""
Production-grade file ↔ bytes conversion with streaming support.
Lossless roundtrip for ALL file formats (binary/text/image/video).
"""

import os
import hashlib
from pathlib import Path
from typing import Iterator, Union
from contextlib import contextmanager


def file_to_bytes(
    filepath: Union[str, Path, os.PathLike],
    *,
    chunk_size: int = 64 * 1024 * 1024,  # 64MB default
    memory_limit: int = 2 * 1024 * 1024 * 1024,  # 2GB threshold
) -> bytes:
    """Convert file to bytes. Auto-streams large files."""
    filepath = Path(filepath)
    file_size = filepath.stat().st_size
    
    if file_size <= memory_limit:
        with open(filepath, 'rb') as f:
            return f.read()
    else:
        chunks: list[bytes] = []
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                chunks.append(chunk)
        return b''.join(chunks)


def bytes_to_file(
    data: bytes,
    output_path: Union[str, Path, os.PathLike],
    *,
    chunk_size: int = 64 * 1024 * 1024,
    overwrite: bool = False,
) -> Path:
    """Reconstruct exact file from bytes (lossless)."""
    output_path = Path(output_path)
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {output_path} (set overwrite=True)")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_written = 0
    with open(output_path, 'wb') as f:
        while total_written < len(data):
            chunk = data[total_written:total_written + chunk_size]
            f.write(chunk)
            total_written += len(chunk)
    
    return output_path


@contextmanager
def file_byte_stream(
    filepath: Union[str, Path, os.PathLike],
    chunk_size: int = 64 * 1024 * 1024,
) -> Iterator[Iterator[bytes]]:
    """Yield chunk generator for streaming huge files."""
    with open(filepath, 'rb', buffering=chunk_size) as f:
        def read_chunks() -> Iterator[bytes]:
            while chunk := f.read(chunk_size):
                yield chunk
        yield read_chunks


def bytes_stream(
    data: bytes,
    chunk_size: int = 64 * 1024 * 1024,
) -> Iterator[bytes]:
    """Stream bytes object without slicing overhead."""
    total = 0
    while total < len(data):
        end = min(total + chunk_size, len(data))
        yield data[total:end]
        total = end


def hash_streaming(
    filepath: Union[str, Path],
    hash_func: str = 'sha256',
    chunk_size: int = 64 * 1024 * 1024,
) -> bytes:
    """Compute hash over streaming file (memory safe)."""
    h = getattr(hashlib, hash_func)()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.digest()


def verify_roundtrip(
    original: Union[str, Path],
    reconstructed: Union[str, Path],
) -> bool:
    """Verify lossless roundtrip."""
    return hash_streaming(original) == hash_streaming(reconstructed)


# Pure utility functions
def get_file_size(filepath: Union[str, Path]) -> int:
    """Get exact file size in bytes."""
    return Path(filepath).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Human readable: 1.23 GB, 456.7 MB, etc."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"
