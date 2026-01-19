# crypto/aead.py
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class AEADContext:
    """
    AES-256-GCM wrapper for encrypting/decrypting data.
    """

    def __init__(self, key: bytes):
        """
        Args:
            key (bytes): 32-byte AES key
        """
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")
        self._key = key
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, plaintext: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        """
        Encrypt data using AES-256-GCM.

        Args:
            plaintext (bytes): Data to encrypt
            nonce (bytes): 12-byte unique nonce for this key
            aad (bytes): Optional additional authenticated data

        Returns:
            bytes: ciphertext with authentication tag
        """
        if len(nonce) != 12:
            raise ValueError("AES-GCM requires a 12-byte nonce")
        return self._aesgcm.encrypt(nonce, plaintext, aad)

    def decrypt(self, ciphertext: bytes, nonce: bytes, aad: bytes = b"") -> bytes:
        """
        Decrypt data using AES-256-GCM.

        Args:
            ciphertext (bytes): Encrypted data
            nonce (bytes): 12-byte nonce used during encryption
            aad (bytes): Optional additional authenticated data

        Returns:
            bytes: plaintext
        """
        if len(nonce) != 12:
            raise ValueError("AES-GCM requires a 12-byte nonce")
        return self._aesgcm.decrypt(nonce, ciphertext, aad)

    @staticmethod
    def generate_nonce() -> bytes:
        """Generate a cryptographically secure 12-byte nonce"""
        return os.urandom(12)
