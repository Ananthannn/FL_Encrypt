#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
import sys

# --- PROJECT ROOT ---
ROOT = Path(__file__).resolve().parents[2]  # points to FL_Encrypt/
sys.path.insert(0, str(ROOT))

from kimura.session.worker import SecureClient
from kimura.protocol.constants import DEFAULT_PORT
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="oqs")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def main():
    # ---- CONFIG ----
    key_path = ROOT / "simulations" / "keys"  # path to your PQC keys
    file_to_send = ROOT / "simulations" / "shared" / "new_file.bin"
    master_host = "127.0.0.1"
    master_port = DEFAULT_PORT

    if not file_to_send.exists():
        logger.error(f"File not found: {file_to_send}")
        return

    # ---- CREATE CLIENT INSTANCE ----
    client = SecureClient(str(key_path), str(file_to_send))

    # ---- CONNECT TO MASTER AND SEND ----
    logger.info(f"Connecting to master at {master_host}:{master_port}")
    try:
        await client.connect_and_send(master_host, master_port)
        logger.info(f"File sent successfully: {file_to_send}")
    except Exception as e:
        logger.error(f"Failed to send file: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker shutdown")
