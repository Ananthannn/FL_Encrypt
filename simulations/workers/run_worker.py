#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
import warnings
import sys
import io
import time
import torch
import wandb
import psutil
import pynvml
from train import train, load_data
from model import get_model
from flwr.common import ndarrays_to_parameters
import pickle
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kimura.session.worker import SecureClient
from kimura.protocol.constants import DEFAULT_PORT
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from kimura.protocol.fl_protocol import FLMessageType, parse_fl_message
# Local cache folder for storing received weights
CACHE_DIR = ROOT / "simulations" / "workers" / "cache_weights"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cached_model_path = CACHE_DIR / "cached_model.pt"

async def training(weights_bytes: bytes, round_no: int) -> bytes:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "data/worker_1/data"
    # ------------------------
    # Initialize monitoring
    # ------------------------
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def get_gpu_stats():
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "gpu_memory_used_MB": mem.used / 1024**2,
            "gpu_utilization_percent": util.gpu,
            "gpu_temperature": temp,
        }
    # ------------------------
    # Deserialize FL message
    # ------------------------
    msg_type, raw_bytes = parse_fl_message(weights_bytes)
    if msg_type != FLMessageType.MODEL_FILE:
        raise ValueError(f"Expected MODEL_FILE message, got {msg_type}")
    # ------------------------
    # Load incoming weights
    # ------------------------
    state_dict = torch.load(io.BytesIO(raw_bytes), map_location=DEVICE)
    model = get_model(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    loader = load_data(DATA_PATH)
    # ------------------------
    # W&B init (once per worker)
    # ------------------------
    if wandb.run is None:
        wandb.init(
            project="federated-kidney-ct",
            name=f"worker_{round_no}",
            config={
                "model": "convnext_large",
                "framework": "kimura_secure_fl",
            }
        )
        wandb.watch(model, log="all", log_freq=20)
    # ------------------------
    # Train
    # ------------------------
    start = time.time()
    model = train(model, loader, DEVICE)
    end = time.time()

    # SECURE AGGREGATION: Add random mask

    # ------------------------
    # System metrics
    # ------------------------
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    gpu = get_gpu_stats()
    samples = len(loader.dataset)
    throughput = samples / (end - start)
    wandb.log({
        "fl_round": round_no,
        "round_time_sec": end - start,
        "samples_per_second": throughput,
        "cpu_usage_percent": cpu,
        "ram_usage_percent": ram,
        **gpu
    })
    # Save CLEAN model to cache (for next round's initial model)
    torch.save(model.state_dict(), cached_model_path)
    params = [p.detach().cpu().numpy() for p in model.state_dict().values()]
    parameters = ndarrays_to_parameters(params)
    num_examples = len(loader.dataset)
    return pickle.dumps((parameters, num_examples))

async def main():
    key_path = ROOT / "simulations" / "keys"
    master_host = "127.0.0.1"
    master_port = DEFAULT_PORT

    client = SecureClient(str(key_path))
    client.set_weights_callback(training)

    # Connect and start FL loop
    # Now the worker will save received model bytes into cached_model_path
    await client.connect_fl(
        master_host, 
        master_port, 
        initial_model_path=str(cached_model_path)
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker shutdown")

"""
It’s still missing many things real research FL systems have:

differential privacy done 

secure aggregation done 

gradient compression

client sampling

heterogeneity handling

Byzantine robustness

"""