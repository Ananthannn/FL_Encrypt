# crypto/keygen.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
from kimura.crypto.mlkem import MLKEM
from kimura.crypto.mldsa import MLDSA
from pathlib import Path


# ========== ML-KEM FUNCTIONS (Server + Client) ==========
def generate_mlkem_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    kem = MLKEM("ML-KEM-768")
    kem_pk, kem_sk = kem.keygen()

    Path(key_dir).mkdir(exist_ok=True)

    with open(Path(key_dir) / "server_mlkem.bin", "wb") as f:
        f.write(len(kem_pk).to_bytes(4, "big"))
        f.write(kem_pk)
        f.write(kem_sk)

    return kem_pk, kem_sk

def generate_mlkem_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    kem = MLKEM("ML-KEM-768")
    kem_pk, kem_sk = kem.keygen()

    Path(key_dir).mkdir(exist_ok=True)

    with open(Path(key_dir) / "client_mlkem.bin", "wb") as f:
        f.write(len(kem_pk).to_bytes(4, "big"))
        f.write(kem_pk)
        f.write(kem_sk)

    return kem_pk, kem_sk

def load_mlkem_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    data = Path(key_dir, "server_mlkem.bin").read_bytes()

    pk_len = int.from_bytes(data[:4], "big")
    kem_pk = data[4:4+pk_len]
    kem_sk = data[4+pk_len:]

    return kem_pk, kem_sk

def load_mlkem_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    data = Path(key_dir, "client_mlkem.bin").read_bytes()

    pk_len = int.from_bytes(data[:4], "big")
    kem_pk = data[4:4+pk_len]
    kem_sk = data[4+pk_len:]

    return kem_pk, kem_sk



# ========== ML-DSA FUNCTIONS (Server + Client) ==========
def generate_mldsa_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    mldsig = MLDSA("ML-DSA-65")
    dsa_pk, dsa_sk = mldsig.keygen()
    Path(key_dir).mkdir(exist_ok=True)
    with open(Path(key_dir) / "server_mldsa.bin", "wb") as f:
        f.write(len(dsa_pk).to_bytes(4, "big"))
        f.write(dsa_pk)
        f.write(dsa_sk)

    return dsa_pk, dsa_sk

def generate_mldsa_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    mldsig = MLDSA("ML-DSA-65")
    dsa_pk, dsa_sk = mldsig.keygen()
    Path(key_dir).mkdir(exist_ok=True)
    with open(Path(key_dir) / "client_mldsa.bin", "wb") as f:
        f.write(len(dsa_pk).to_bytes(4, "big"))
        f.write(dsa_pk)
        f.write(dsa_sk)

    return dsa_pk, dsa_sk

def load_mldsa_server_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    data = Path(key_dir, "server_mldsa.bin").read_bytes()
    pk_len = int.from_bytes(data[:4], "big")
    dsa_pk = data[4:4+pk_len]
    dsa_sk = data[4+pk_len:]

    return dsa_pk, dsa_sk

def load_mldsa_client_keys(key_dir: str = "keys") -> tuple[bytes, bytes]:
    data = Path(key_dir, "client_mldsa.bin").read_bytes()
    pk_len = int.from_bytes(data[:4], "big")
    dsa_pk = data[4:4+pk_len]
    dsa_sk = data[4+pk_len:]

    return dsa_pk, dsa_sk
