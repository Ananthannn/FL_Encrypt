import hashlib

def sha256(data: bytes) -> bytes:
    """
    Compute SHA-256 hash of input data.
    Returns:
        32-byte digest
    """
    digest = hashlib.sha256(data).digest()  # returns bytes
    return digest
def sha512(data: bytes) -> bytes:
    """
    Compute SHA-512 hash of input data.
    Returns:
        64-byte digest
    """
    return hashlib.sha512(data).digest()  # 64 bytes