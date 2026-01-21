import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
import oqs
import os
import sys
from kimura.crypto import aead
from kimura.crypto.aead import AEADContext
from typing import Tuple, Optional, Union
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class MLKEM:
    """
    ML-KEM (Kyber successor, NIST PQC standard) wrapper using liboqs-python.
    
    Supports ML-KEM-512, ML-KEM-768, ML-KEM-1024.
    Provides keygen, encrypt/decrypt (hybrid via shared secret), encaps/decaps.
    
    Usage:
        kem = MLKEM("ML-KEM-512")
        public_key, secret_key = kem.keygen()
        ciphertext, shared_secret = kem.encaps(public_key)
        decrypted_ss = kem.decaps(ciphertext, secret_key)
        nonce, ciphertext, tag = kem.encrypt(b"message", shared_secret)
        plaintext = kem.decrypt(nonce, ciphertext, tag, shared_secret)
    """
    
    def __init__(self, alg: str = "ML-KEM-512"):
        """
        Initialize with ML-KEM algorithm.
        
        :param alg: One of "ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"
        """
        if alg not in oqs.get_enabled_kem_mechanisms():
            raise ValueError(f"Unsupported algorithm: {alg}. Available: {oqs.get_enabled_kem_mechanisms()}")
        self.alg = alg
        self.kem = oqs.KeyEncapsulation(alg)
        self.length_public_key = self.kem.length_public_key
        self.length_secret_key = self.kem.length_secret_key
        self.length_ciphertext = self.kem.length_ciphertext
        self.length_shared_secret = self.kem.length_shared_secret
        
    def keygen(self) -> Tuple[bytes, bytes]:
        """
        Key generation: Generate public and secret keypair.
        
        :return: (public_key, secret_key)
        """
        public_key = self.kem.generate_keypair()
        secret_key = self.kem.export_secret_key()
        return public_key, secret_key
    
    def load_keypair(self, public_key: bytes, secret_key: bytes) -> None:
        """
        Load existing keypair (for resumption).
        """
        if len(public_key) != self.length_public_key or len(secret_key) != self.length_secret_key:
            raise ValueError("Invalid key lengths")
        # Note: liboqs-python context is stateful; for production, consider serializing properly
        self.kem = oqs.KeyEncapsulation(self.alg, secret_key)
    
    def encaps(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate: Generate shared secret and ciphertext using public key.
        
        :param public_key: Recipient's public key
        :return: (ciphertext, shared_secret)
        """
        ciphertext, shared_secret = self.kem.encap_secret(public_key)
        return ciphertext, shared_secret
    
    def decaps(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulate: Recover shared secret from ciphertext using secret key.
        
        :param ciphertext: Encapsulator's ciphertext
        :param secret_key: Recipient's secret key
        :return: shared_secret
        """
        if len(secret_key) != self.length_secret_key:
            raise ValueError("Invalid secret key length")
        temp_kem = oqs.KeyEncapsulation(self.alg, secret_key)
        shared_secret = temp_kem.decap_secret(ciphertext)
        return shared_secret

    # inside MLKEM
    def encrypt(self, plaintext: bytes, shared_secret: bytes, aad: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
            Hybrid decrypt: AES-256-GCM using shared secret.
            
            :param nonce: IV/nonce
            :param ciphertext: Encrypted message
            :param tag: Authentication tag
            :param shared_secret: ML-KEM shared secret
            :param aad: Associated data (must match encrypt)
            :return: plaintext
        """
        from crypto.aead import AEADContext  # import your wrapper
        nonce = AEADContext.generate_nonce()
        aead = AEADContext(shared_secret)
        ciphertext = aead.encrypt(plaintext, nonce, aad or b"")
        return nonce, ciphertext

    def decrypt(self, nonce: bytes, ciphertext: bytes, shared_secret: bytes, aad: Optional[bytes] = None) -> bytes:
        """
            Hybrid decrypt: AES-256-GCM using shared secret.
            
            :param nonce: IV/nonce
            :param ciphertext: Encrypted message
            :param tag: Authentication tag
            :param shared_secret: ML-KEM shared secret
            :param aad: Associated data (must match encrypt)
            :return: plaintext
        """
        if len(shared_secret) != 32:
            raise ValueError("Shared secret must be 32 bytes")
        aead = AEADContext(shared_secret)
        plaintext = aead.decrypt(ciphertext, nonce, aad or b"")
        return plaintext