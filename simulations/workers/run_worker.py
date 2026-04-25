#!/usr/bin/env python3

import asyncio
import json
import json
import logging
from pathlib import Path
import warnings
import sys
import io
import time
import lz4.frame
import torch
import wandb
import psutil
import pynvml
import model
from train import DEVICE, train, load_data
from model import get_model
import signal
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kimura.session.worker import SecureClient
from kimura.protocol.constants import DEFAULT_PORT
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from kimura.protocol.fl_protocol import FLMessageType, parse_fl_message

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
    decompressed = lz4.frame.decompress(raw_bytes)
    incoming_state_dict = {k: v.float() for k, v in torch.load(io.BytesIO(decompressed), map_location=DEVICE).items()}
    # Strip _module. prefix if present
    if list(incoming_state_dict.keys())[0].startswith("_module."):
        incoming_state_dict = {k.replace("_module.", ""): v for k, v in incoming_state_dict.items()}
    model = get_model(DEVICE)
    model.load_state_dict(incoming_state_dict, strict=False)
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
    # ------------------------
    # Compute ΔW relative to the server model just received
    # ------------------------
    old_state_dict = incoming_state_dict
    new_state_dict = model.state_dict()
    # Compute delta only on keys present in both
    common_keys = old_state_dict.keys() & new_state_dict.keys()
    # Keep full precision CPU tensors for deltas
    delta_weights = {k: new_state_dict[k].cpu() - old_state_dict[k].cpu() for k in common_keys}
    # Include number of training samples
    payload_dict = {
        "delta": delta_weights,
        "num_examples": len(loader.dataset)  # actual number of samples this worker trained on
    }
    # Serialize + compress
    buffer = io.BytesIO()
    torch.save(payload_dict, buffer)
    payload_bytes = lz4.frame.compress(buffer.getvalue())
    return payload_bytes

async def main():
    key_path = ROOT / "simulations" / "keys"
    master_host = "192.168.50.1"
    master_port = DEFAULT_PORT

    client = SecureClient(str(key_path))
    client.set_weights_callback(training)

    # Setup shutdown event
    shutdown_event = asyncio.Event()

    # Define signal handlers for SIGINT/SIGTERM
    def handle_exit(*args):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_exit)

    # Connect and start FL loop
    fl_task = asyncio.create_task(client.connect_fl(master_host, master_port))

    # Wait until shutdown signal
    await shutdown_event.wait()

    logger.info("Graceful shutdown starting...")
    fl_task.cancel()
    try:
        await fl_task
    except asyncio.CancelledError:
        logger.info("FL loop cancelled cleanly")

    # Optional: cleanup wandb
    if wandb.run is not None:
        wandb.finish()
        logger.info("W&B run finished")

    # Optional: cleanup GPU monitoring
    pynvml.nvmlShutdown()
    logger.info("Worker shutdown complete")

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
