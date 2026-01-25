#!/usr/bin/env python3
import asyncio
import logging
from pathlib import Path
import sys
import json
import numpy as np
import io
import warnings


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
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


# 🆕 GLOBAL ORCHESTRATOR (shared across tasks)
orchestrator = None


async def main():
    global orchestrator
    
    # ---- CREATE SERVER + ORCHESTRATOR ----
    server = SecureServer(str(KEY_PATH), base_output=str(RECEIVED_DIR))
    
    # 🔥 CRITICAL: DISABLE KIMURA AUTO-BEHAVIOR
    server.auto_broadcast_on_ready = False  # No auto-model send
    server.auto_training_loop = False       # No auto-FL cycle
    
    orchestrator = Orchestrator(server=server, state_path=STATE_PATH, model_path=MODEL_PATH)
    
    # ---- STATUS MONITOR (every 5s) ----
    async def status_monitor():
        while True:
            workers_connected = len(orchestrator.workers)
            ready_workers = sum(1 for st in orchestrator.workers.values() if st == WorkerState.READY)
            logger.info(f"[MASTER] Workers: {workers_connected} | READY: {ready_workers} | State: {orchestrator.master_state.value} | Round: {orchestrator.round}")
            await asyncio.sleep(5)
    
    # 🆕 TRAINING TRIGGER SERVER (localhost:8444)
    async def training_trigger_server():
        tcp_server = await asyncio.start_server(trigger_handler, '127.0.0.1', 8444)
        logger.info("TRAINING TRIGGER listening on localhost:8444")
        async with tcp_server:
            await tcp_server.serve_forever()
    
    async def trigger_handler(reader, writer):
        data = await reader.read(1024)
        cmd = data.decode().strip()
        logger.info(f"TRAINING TRIGGER received: {cmd}")
        
        try:
            if cmd.startswith("START"):
                rounds = int(cmd.split()[1])
                await orchestrator.start_training(rounds=rounds, min_workers=1)
                logger.info(f"✅ Training completed: {rounds} rounds")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    # ---- FIXED CALLBACKS - Handle RAW NPZ bytes ----
    async def on_result_received(client_id, data_bytes):
        """Handle RAW .npz bytes from workers (not JSON)"""
        global orchestrator
        
        if not data_bytes:
            logger.warning(f"Worker {client_id} sent empty data")
            return
        
        try:
            # 🆕 Save RAW NPZ directly
            out_path = RECEIVED_DIR / f"worker_{client_id}_update_round{orchestrator.round}.npz"
            with open(out_path, "wb") as f:
                f.write(data_bytes)
            logger.info(f"💾 Saved RAW update from Worker {client_id} | Size: {len(data_bytes)/1024:.1f}KB")
            
            # Notify orchestrator
            await orchestrator.receive_result(client_id, data_bytes, orchestrator.round)
            
            # Auto-aggregate when all workers done
            if orchestrator.all_results_received():
                await aggregate_and_broadcast(orchestrator.round)
                
        except Exception as e:
            logger.error(f"❌ Error processing {client_id}: {e}")
    
    async def aggregate_and_broadcast(round_no):
        """Aggregate all worker updates and broadcast average"""
        updates = []
        
        # Load all updates for this round
        for f in RECEIVED_DIR.glob(f"worker_*_update_round{round_no}.npz"):
            arr = np.load(f, allow_pickle=True)
            updates.append({k: arr[k] for k in arr.files})
            logger.info(f"📊 Loaded update from {f.name}")
        
        if not updates:
            logger.warning(f"No updates found for round {round_no}")
            return
        
        # Average weights across workers
        keys = updates[0].keys()
        avg_weights = {k: np.mean([u[k] for u in updates], axis=0) for k in keys}
        
        # Serialize averaged model
        buf = io.BytesIO()
        np.savez(buf, **avg_weights)
        buf.seek(0)
        
        # Broadcast to all workers
        await server.broadcast_weights(buf.getvalue())
        logger.info(f"📡 Round {round_no} → Broadcasted avg weights to {len(orchestrator.workers)} workers")
    
    # Hook FIXED callbacks
    server.on_worker_connected = orchestrator.on_worker_connected
    server.on_worker_ready = orchestrator.on_worker_ready
    server.on_result_received = on_result_received
    
    logger.info("🎬 Master booting - Workers will STAY IDLE until trigger...")
    
    # 🔥 START ALL SERVICES TOGETHER
    await asyncio.gather(
        server.serve_forever(DEFAULT_PORT),     # Workers connect on 8443
        status_monitor(),                       # Status every 5s
        training_trigger_server()               # Trigger on 8444
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Master shutdown")
