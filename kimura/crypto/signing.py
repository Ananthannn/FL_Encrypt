import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
from crypto.mldsa import MLDSA
from crypto.keygen import (
    generate_mlkem_server_keys, generate_mlkem_client_keys,
    generate_mldsa_server_keys, generate_mldsa_client_keys,
    load_mlkem_server_keys, load_mlkem_client_keys,
    load_mldsa_server_keys, load_mldsa_client_keys
)
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
    dsa = MLDSA("ML-DSA-65")
    return dsa.sign(message, secret_key)

def verify_message(message: bytes, signature: bytes, public_key: Optional[bytes] = None) -> bool:
    if not public_key:
        raise ValueError("ML-DSA public key not provided")
    dsa = MLDSA("ML-DSA-65")
    return dsa.verify(message, signature, public_key)