#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[2]  # adjust if your script is nested differently
sys.path.insert(0, str(ROOT))  # now Python can find shared.state etc.
import warnings
from kimura.session.master import SecureServer
from kimura.protocol.constants import DEFAULT_PORT
from orchestrator import Orchestrator
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# Paths
KEY_PATH = ROOT / "simulations" / "keys"
MASTER_DIR = ROOT / "simulations" / "master"
INITIAL_MODEL_PATH = MASTER_DIR / "initial_global" / "initial_global.pt"
GLOBAL_MODEL_PATH = MASTER_DIR / "global_w" / "global_w.pt"
STATE_PATH = MASTER_DIR / "master_state.json"
# Ensure directories exist
GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
INITIAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

async def main():
    if not INITIAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Initial global model not found: {INITIAL_MODEL_PATH}"
    )
    # ---- CREATE SERVER ----
    server = SecureServer(str(KEY_PATH), base_output=str(MASTER_DIR))

    # ---- CREATE ORCHESTRATOR ----
    orchestrator = Orchestrator(
        server=server,
        state_path=STATE_PATH,
        initial_model_path=INITIAL_MODEL_PATH,
        global_model_path=GLOBAL_MODEL_PATH
    )
    # ---- STATUS PRINTING ----
    def print_status():
        logger.info(f"[MASTER] Workers connected: {len(orchestrator.workers)}")
        logger.info(f"[MASTER] MasterState: {orchestrator.master_state.value}")


    async def on_result_received(client_id, data_bytes):
        try:
            payload = json.loads(data_bytes)
            round_no = payload["round_no"]
            weights = bytes.fromhex(payload["weights"])
        except Exception as e:
            logger.error(f"Bad update from {client_id}: {e}")
            return

        # Forward to orchestrator ONLY
        await orchestrator.receive_result(client_id, weights, round_no)

    async def control_server(orchestrator):
        async def handle(reader, writer):
            line = await reader.readline()
            cmd = line.decode().strip()

            if cmd.startswith("START"):
                # Instead of checking master_state, check training_active
                if orchestrator.training_active:
                    writer.write(b"BUSY\n")
                else:
                    _, rounds = cmd.split()
                    asyncio.create_task(
                        orchestrator.start_training(rounds=int(rounds), min_workers=1)
                    )
                    writer.write(b"OK\n")

            else:
                writer.write(b"UNKNOWN\n")

            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 8444)
        logging.info("[CONTROL] Listening on 127.0.0.1:8444")

        async with server:
            await server.serve_forever()


        server = await asyncio.start_server(handle, "127.0.0.1", 8444)
        logger.info("[CONTROL] Listening on 127.0.0.1:8444")

        async with server:
            await server.serve_forever()

    # Hook callbacks
    server.on_worker_connected = orchestrator.on_worker_connected
    server.on_worker_ready = orchestrator.on_worker_ready
    server.on_result_received = on_result_received

    logger.info("Master booting... ")
    print_status()
    await asyncio.gather(
        orchestrator.run(port=DEFAULT_PORT),
        control_server(orchestrator),
    )
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Master shutdown")
