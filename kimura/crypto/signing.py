import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
from kimura.crypto.mldsa import MLDSA
from kimura.crypto.keygen import (
    generate_mlkem_server_keys, generate_mlkem_client_keys,
    generate_mldsa_server_keys, generate_mldsa_client_keys,
    load_mlkem_server_keys, load_mlkem_client_keys,
    load_mldsa_server_keys, load_mldsa_client_keys
)
from cryptography.hazmat.primitives import serialization
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
from pathlib import Path
from typing import Optional, Tuple
import hashlib

def ensure_keys_exist(base_dir: str = "keys", role: str = "server") -> Tuple[bytes, bytes, bytes, bytes]:
    """
    Ensure ML-KEM + ML-DSA keys exist for the given role (server/client).
    Creates folders and generates keys if missing.

    Returns: (kem_pk, kem_sk, dsa_pk, dsa_sk)
    """
    if role not in ("server", "client"):
        raise ValueError("role must be 'server' or 'client'")

    role_dir = Path(base_dir)
    role_dir.mkdir(parents=True, exist_ok=True)
    kem_file = role_dir / "mlkem.pem"
    dsa_file = role_dir / "mldsa.pem"
    missing = []
    if not kem_file.exists():
        missing.append("ML-KEM")
        if role == "server":
            generate_mlkem_server_keys(str(role_dir))
        else:
            generate_mlkem_client_keys(str(role_dir))
    if not dsa_file.exists():
        missing.append("ML-DSA")
        if role == "server":
            generate_mldsa_server_keys(str(role_dir))
        else:
            generate_mldsa_client_keys(str(role_dir))
    # Load keys
    if role == "server":
        kem_pk, kem_sk = load_mlkem_server_keys(str(role_dir))
        dsa_pk, dsa_sk = load_mldsa_server_keys(str(role_dir))
    else:
        kem_pk, kem_sk = load_mlkem_client_keys(str(role_dir))
        dsa_pk, dsa_sk = load_mldsa_client_keys(str(role_dir))
    return kem_pk, kem_sk, dsa_pk, dsa_sk

def hash_file(file_path: Path) -> bytes:
    """Compute SHA-256 hash of entire file for integrity."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.digest()

def sign_message(message: bytes, secret_key: bytes) -> bytes:
    if not secret_key:
        raise ValueError("ML-DSA secret key not provided")

    if b"-----BEGIN" in secret_key:
        loaded = serialization.load_pem_private_key(secret_key, password=None)
        secret_key = loaded.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )

    dsa = MLDSA("ML-DSA-65")
    return dsa.sign(message, secret_key)


def verify_message(message: bytes, signature: bytes, public_key: Optional[bytes] = None) -> bool:
    if not public_key:
        raise ValueError("ML-DSA public key not provided")

    # If the public key is PEM, convert to raw bytes first
    if b"-----BEGIN" in public_key:
        loaded = serialization.load_pem_public_key(public_key)
        public_key = loaded.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    dsa = MLDSA("ML-DSA-65")
    return dsa.verify(message, signature, public_key)

def verify_peer_key(peer_key: bytes, key_path: str, role: str, ephemeral: bool = False) -> None:
    """
    Trust-On-First-Use (TOFU) check for a peer's public key.

    Args:
        peer_key: raw bytes of the peer's public key.
        key_path: directory to store fingerprint.
        role: "client" or "server" (for file naming)
        ephemeral: if True, skip fingerprint verification (ephemeral session)
    
    Raises:
        ProtocolError if key changes on subsequent connections.
    """
    # Compute SHA-256 fingerprint
    fingerprint = hashlib.sha256(peer_key).hexdigest()

    if ephemeral:
        logger.info(f"[{role.upper()}] Ephemeral session: skipping fingerprint check. Fingerprint would be {fingerprint}")
        return

    # Ensure secure directory exists
    key_dir = Path(key_path)
    key_dir.mkdir(parents=True, exist_ok=True)
    key_dir.chmod(0o700)

    # Fingerprint file location
    fp_file = key_dir / f"{role}_peer.fingerprint"

    if fp_file.exists():
        stored_fp = fp_file.read_text().strip()
        if stored_fp != fingerprint:
            raise ValueError(
                f"Peer {role} key mismatch! Possible MITM attack.\n"
                f"Expected: {stored_fp}\nGot:      {fingerprint}"
            )
    else:
        # First use: securely write fingerprint
        fd = os.open(fp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(fingerprint)
        logger.info(f"TOFU: Stored new {role} peer fingerprint {fingerprint}")


def vverify_peer_key(peer_key: bytes, key_path: str, role: str) -> None:
    """
    Trust-On-First-Use (TOFU) check for a peer's public key.

    Args:
        peer_key: raw bytes of the peer's public key.
        key_path: directory to store fingerprint.
        role: "client" or "server" (for file naming)
    
    Raises:
        ProtocolError if key changes on subsequent connections.
    """
    # Compute SHA-256 fingerprint
    fingerprint = hashlib.sha256(peer_key).hexdigest()

    # Ensure secure directory exists
    key_dir = Path(key_path)
    key_dir.mkdir(parents=True, exist_ok=True)
    # Make directory readable/writeable only by owner
    key_dir.chmod(0o700)

    # Fingerprint file location
    fp_file = key_dir / f"{role}_peer.fingerprint"

    if fp_file.exists():
        stored_fp = fp_file.read_text().strip()
        if stored_fp != fingerprint:
            raise ValueError(
                f"Peer {role} key mismatch! Possible MITM attack.\n"
                f"Expected: {stored_fp}\nGot:      {fingerprint}"
            )
    else:
        # First use: securely write fingerprint
        # Use os.open with mode 0o600 to set permissions atomically
        fd = os.open(fp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(fingerprint)
        logger.info(f"TOFU: Stored new {role} peer fingerprint {fingerprint}")

