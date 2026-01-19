# crypto/kdf.py
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def hkdf_sha256(secret: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """
    Derive cryptographic keys using HKDF-SHA256.

    Args:
        secret (bytes): Input keying material (IKM) — e.g., shared secret from KEM.
        salt (bytes): Non-secret random value, used to prevent rainbow attacks.
        info (bytes): Context/application-specific info, e.g., b'pqc-p2p-session'.
        length (int): Number of bytes to derive (e.g., 32 for AES-256 key).

    Returns:
        bytes: Derived key of requested length.
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info
    )
    return hkdf.derive(secret)