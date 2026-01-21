#!/usr/bin/env python3
import asyncio
import logging
import sys
from pathlib import Path

# --- PROJECT ROOT ---
ROOT = Path(__file__).resolve().parents[2]  # FL_Encrypt/
sys.path.insert(0, str(ROOT))  # <- add this so imports work

output_path = ROOT / "simulations" / "shared" / "received_from_workers.pt"
output_path.parent.mkdir(parents=True, exist_ok=True)

from kimura.session.master import SecureServer  # your PQC + TCP Master
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
    key_path = ROOT / "kimura" / "keys"  # absolute path to your keys
    output_path = ROOT / "simulations" / "shared" / "received_from_workers.pt"

    # ---- CREATE MASTER INSTANCE ----
    master = SecureServer(str(key_path), base_output=str(output_path))
    logger.info("Master booting... waiting for workers to connect")

    # ---- START SERVER TUNNEL ----
    await master.serve_forever(port=DEFAULT_PORT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Master shutdown")