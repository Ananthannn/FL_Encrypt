# crypto/keygen.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
from crypto.mlkem import MLKEM
from crypto.mldsa import MLDSA
from pathlib import Path

# ========== ML-KEM FUNCTIONS (Server + Client) ==========
def generate_mlkem_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Generate ML-KEM keys for SERVER only."""
    kem = MLKEM("ML-KEM-768")
    kem_pk, kem_sk = kem.keygen()
    Path(key_dir).mkdir(exist_ok=True)
    key_data = len(kem_pk).to_bytes(4, "big") + kem_pk + kem_sk
    with open(Path(key_dir) / "server_mlkem.pem", "wb") as f:
        f.write(key_data)
    return kem_pk, kem_sk

def generate_mlkem_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Generate ML-KEM keys for CLIENT only."""
    kem = MLKEM("ML-KEM-768")
    kem_pk, kem_sk = kem.keygen()
    Path(key_dir).mkdir(exist_ok=True)
    key_data = len(kem_pk).to_bytes(4, "big") + kem_pk + kem_sk
    with open(Path(key_dir) / "client_mlkem.pem", "wb") as f:
        f.write(key_data)
    return kem_pk, kem_sk

def load_mlkem_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Load ML-KEM server keys."""
    with open(Path(key_dir) / "server_mlkem.pem", "rb") as f:
        pk_len = int.from_bytes(f.read(4), "big")
        kem_pk = f.read(pk_len)
        kem_sk = f.read()
    return kem_pk, kem_sk

def load_mlkem_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Load ML-KEM client keys."""
    with open(Path(key_dir) / "client_mlkem.pem", "rb") as f:
        pk_len = int.from_bytes(f.read(4), "big")
        kem_pk = f.read(pk_len)
        kem_sk = f.read()
    return kem_pk, kem_sk

# ========== ML-DSA FUNCTIONS (Server + Client) ==========
def generate_mldsa_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Generate ML-DSA keys for SERVER (to verify client signatures)."""
    mldsig = MLDSA("ML-DSA-65")
    dsa_pk, dsa_sk = mldsig.keygen()
    Path(key_dir).mkdir(exist_ok=True)
    key_data = len(dsa_pk).to_bytes(4, "big") + dsa_pk + dsa_sk
    with open(Path(key_dir) / "server_mldsa.pem", "wb") as f:
        f.write(key_data)
    return dsa_pk, dsa_sk

def generate_mldsa_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Generate ML-DSA keys for CLIENT (to verify server signatures)."""
    mldsig = MLDSA("ML-DSA-65")
    dsa_pk, dsa_sk = mldsig.keygen()
    Path(key_dir).mkdir(exist_ok=True)
    key_data = len(dsa_pk).to_bytes(4, "big") + dsa_pk + dsa_sk
    with open(Path(key_dir) / "client_mldsa.pem", "wb") as f:
        f.write(key_data)
    return dsa_pk, dsa_sk

def load_mldsa_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Load ML-DSA server keys."""
    with open(Path(key_dir) / "server_mldsa.pem", "rb") as f:
        pk_len = int.from_bytes(f.read(4), "big")
        dsa_pk = f.read(pk_len)
        dsa_sk = f.read()
    return dsa_pk, dsa_sk

def load_mldsa_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    """Load ML-DSA client keys."""
    with open(Path(key_dir) / "client_mldsa.pem", "rb") as f:
        pk_len = int.from_bytes(f.read(4), "big")
        dsa_pk = f.read(pk_len)
        dsa_sk = f.read()
    return dsa_pk, dsa_sk