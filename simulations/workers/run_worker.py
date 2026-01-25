#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
import warnings
import numpy as np
import io
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from kimura.session.worker import SecureClient
from kimura.protocol.constants import DEFAULT_PORT
# we do not train the model here by loading it from model.py with local dataset and from here the weight is send
"""✅ Wiring everything together
✅ Register callback
✅ Logging (W&B) we will only be using this for the logging purpose not for training This is where weights_callback belongs
"""
async def fake_training(weights_bytes: bytes, round_no: int) -> bytes:
    # Load model
    buffer = io.BytesIO(weights_bytes)
    arr = np.load(buffer, allow_pickle=True)

    # Perform fake "training"
    updated = {k: arr[k] * (1 + 0.01 * (round_no + 1)) for k in arr.files}  # example per-round scaling

    # Serialize back
    out = io.BytesIO()
    np.savez(out, **updated)
    
    # Reset pointer to start BEFORE sending
    out.seek(0)
    
    return out.getvalue()



async def main():
    key_path = ROOT / "simulations" / "keys"
    master_host = "127.0.0.1"
    master_port = DEFAULT_PORT
    initial_model_path = ROOT / "simulations" / "shared" / "model.npz"

    client = SecureClient(str(key_path))
    client.set_weights_callback(fake_training)
    # Connect and start FL loop
    await client.connect_fl(master_host, master_port, initial_model_path=str(initial_model_path))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker shutdown")
