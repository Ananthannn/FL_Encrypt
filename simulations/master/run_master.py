#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # adjust if your script is nested differently
sys.path.insert(0, str(ROOT))  # now Python can find shared.state etc.
import warnings
import numpy as np
import io
import json
from kimura.session.master import SecureServer
from kimura.protocol.constants import DEFAULT_PORT
from orchestrator import Orchestrator
from shared.state import WorkerState
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
KEY_PATH = ROOT / "simulations" / "keys"
MODEL_PATH = ROOT / "simulations" / "shared" / "model.npz"
RECEIVED_DIR = ROOT / "simulations" / "shared"
RECEIVED_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = RECEIVED_DIR / "master_state.json"

async def main():
    # ---- CREATE SERVER ----
    server = SecureServer(str(KEY_PATH), base_output=str(RECEIVED_DIR))

    # ---- CREATE ORCHESTRATOR ----
    orchestrator = Orchestrator(
        server=server,
        state_path=STATE_PATH,
        model_path=MODEL_PATH
    )
    # ---- STATUS PRINTING ----
    def print_status():
        num_workers = len(orchestrator.workers)

        if not orchestrator.workers:
            state = "IDLE"
        elif any(st == WorkerState.HANDSHAKE_DONE for st in orchestrator.workers.values()):
            # Workers are ready but haven't started training
            state = "WAITING_FOR_TASK"
        elif any(st in (WorkerState.TASK_SENT, WorkerState.TRAINING) for st in orchestrator.workers.values()):
            state = "ROUND_IN_PROGRESS"
        elif orchestrator.all_results_received():
            state = "AGGREGATING"
        elif any(st == WorkerState.RESULT_RECEIVED for st in orchestrator.workers.values()):
            state = "AGGREGATING"
        else:
            state = "IDLE"

        logger.info(f"[MASTER] Workers connected: {num_workers}")
        logger.info(f"[MASTER] State: {state}")

    # ---- CALLBACK: Receive updates ----
    async def on_result_received(client_id, data_bytes):
        # client_id is worker_id (string) from handshake
        # Reject if handshake not complete
        if client_id not in orchestrator.workers or orchestrator.workers[client_id] not in (
            WorkerState.HANDSHAKE_DONE,
            WorkerState.TASK_SENT
        ):
            logger.error(f"Worker {client_id} sent result before handshake or task dispatch!")
            return

        # Decode JSON payload
        payload = json.loads(data_bytes)
        round_no = payload["round_no"]
        weights = bytes.fromhex(payload["weights"])

        # Save update to disk
        out_path = RECEIVED_DIR / f"worker_{client_id}_update_round{round_no}.npz"
        with open(out_path, "wb") as f:
            f.write(weights)
        logger.info(f"Saved update from Worker {client_id} for round {round_no} to {out_path}")

        # Tell orchestrator the result arrived
        await orchestrator.receive_result(client_id, weights, round_no)

        # Check if all results received before aggregation
        if orchestrator.all_results_received():
            # Aggregate updates
            updates = []
            for f in RECEIVED_DIR.glob(f"worker_*_update_round{round_no}.npz"):
                arr = np.load(f, allow_pickle=True)
                updates.append({k: arr[k] for k in arr.files})
                logger.info(f"Loaded update from {f}")
            if updates:
                keys = updates[0].keys()
                avg_update = {k: sum(u[k] for u in updates) / len(updates) for k in keys}

                # Serialize and broadcast
                buf = io.BytesIO()
                np.savez(buf, **avg_update)
                buf.seek(0)
                await server.broadcast_weights(buf.getvalue())
                logger.info(f"Broadcasted averaged weights for round {round_no} to all clients")


    # Hook callbacks
    server.on_worker_connected = orchestrator.on_worker_connected
    server.on_worker_ready = orchestrator.on_worker_ready
    server.on_result_received = on_result_received

    logger.info("Master booting... ")
    print_status()
    await orchestrator.run(port=DEFAULT_PORT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Master shutdown")
