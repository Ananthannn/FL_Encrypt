import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
import oqs
import os
from cryptography.hazmat.primitives import serialization
from typing import Tuple, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
import logging
logger = logging.getLogger(__name__)

class MLDSA:
    """
    Production-level ML-DSA (Dilithium successor, NIST PQC standard) wrapper using liboqs-python.
    
    Supports ML-DSA-44, ML-DSA-65, ML-DSA-87.
    Provides keygen, sign, verify. Bytes-only API with optional hybrid RSA fallback.
    
    Usage:
        sig = MLDSA("ML-DSA-65")
        public_key, secret_key = sig.keygen()
        signature = sig.sign(b"message", secret_key)
        valid = sig.verify(b"message", signature, public_key)
    """
    
    def __init__(self, alg: str = "ML-DSA-65"):
        """
        Initialize with ML-DSA algorithm.
        
        :param alg: One of "ML-DSA-44", "ML-DSA-65", "ML-DSA-87"
        """
        if alg not in oqs.get_enabled_sig_mechanisms():
            raise ValueError(f"Unsupported algorithm: {alg}. Available: {oqs.get_enabled_sig_mechanisms()}")
        self.alg = alg
        self.sig = oqs.Signature(alg)
        self.length_public_key = self.sig.length_public_key
        self.length_secret_key = self.sig.length_secret_key
        self.max_signature_len = self.sig.length_signature
        
    def keygen(self) -> Tuple[bytes, bytes]:
        """Key generation: Generate public and secret keypair."""
        temp_sig = oqs.Signature(self.alg)  # Fresh instance
        public_key = temp_sig.generate_keypair()
        secret_key = temp_sig.export_secret_key()
        return public_key, secret_key
    
    def load_keypair(self, public_key: bytes, secret_key: bytes) -> None:
        """
        Load existing keypair.
        """
        if len(public_key) != self.length_public_key or len(secret_key) != self.length_secret_key:
            raise ValueError("Invalid key lengths")
        self.sig = oqs.Signature(self.alg, secret_key)
    
    def sign(self, message: bytes, secret_key: bytes) -> bytes:
        """"
        Sign message using secret key.
        
        :param message: Bytes to sign
        :param secret_key: Signer's secret key
        :return: signature
        """
        if len(secret_key) != self.length_secret_key:
            raise ValueError("Invalid secret key length")
        signer = oqs.Signature(self.alg, secret_key)
        return signer.sign(message)

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verify signature against message and public key.
        Accepts either RAW ML-DSA bytes or PEM-encoded public keys.
        """

        # === 🔹 STEP 1: If it's PEM, convert it to RAW ===
        if isinstance(public_key, bytes) and b"-----BEGIN" in public_key:
            try:
                loaded = serialization.load_pem_public_key(public_key)
                public_key = loaded.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw
                )
            except Exception as e:
                raise ValueError(f"Failed to parse PEM public key: {e}")

        # === 🔹 STEP 2: Now enforce correct length ===
        if len(public_key) != self.length_public_key:
            raise ValueError(
                f"Invalid public key length: got {len(public_key)}, "
                f"expected {self.length_public_key}"
            )

        # === 🔹 STEP 3: Verify ===
        sig = oqs.Signature(self.alg)
        try:
            sig.verify(message, signature, public_key)
            return True
        except oqs.SignatureError:
            return False
    
    def hybrid_sign(self, message: bytes, rsa_private_key, pqc_private_key: bytes) -> bytes:
        """
        Hybrid sign: ML-DSA + RSA-PSS (for max security).
        
        :param message: Bytes to sign
        :param rsa_private_key: cryptography RSA private key
        :param pqc_private_key: ML-DSA secret key
        :return: concatenated (ml_dsa_sig || rsa_sig)
        """
        ml_dsa_sig = self.sign(message, pqc_private_key)
        rsa_sig = rsa_private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return ml_dsa_sig + rsa_sig
    
    @staticmethod
    def hybrid_verify(message: bytes, hybrid_sig: bytes, ml_dsa_public_key: bytes, rsa_public_key) -> bool:
        """
        Verify hybrid signature.
        """
        ml_dsa_len = MLDSA("ML-DSA-65").max_signature_len  # Fixed for example
        ml_dsa_sig = hybrid_sig[:ml_dsa_len]
        rsa_sig = hybrid_sig[ml_dsa_len:]
        
        # Verify ML-DSA
        sig = MLDSA("ML-DSA-65")
        if not sig.verify(message, ml_dsa_sig, ml_dsa_public_key):
            return False
        
        # Verify RSA
        try:
            rsa_public_key.verify(
                rsa_sig,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False