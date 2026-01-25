#!/usr/bin/env python3
"""
Orchestrator (Master Brain)

Responsibilities:
- Track worker lifecycle
- Dispatch tasks
- Receive results
- Aggregate outputs
- Maintain state consistency

Crypto, handshake, transport are handled by SecureServer.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any
from shared.state import MasterState, WorkerState
logger = logging.getLogger("orchestrator")
from simulations.master.aggregator import aggregate
# ============================================================
# ORCHESTRATOR CORE
# ============================================================
class Orchestrator:
    def __init__(self, server, state_path: Path | None = None, model_path: Path | None = None):
        """
        Args:
            server: SecureServer instance (crypto + TCP already inside)
            state_path: where master state.json is stored
        """
        self.server = server

        self.workers: Dict[str, WorkerState] = {}
        self.results: Dict[str, bytes] = {}
        self.round: int = 0
        self.model_path = model_path
        self.training_active = False
        self.enable_resume = False 
        self.master_state = MasterState.IDLE

        self.state_path = state_path
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_state()

        logger.info("Orchestrator initialized")

    # ========================================================
    # STATE MANAGEMENT
    # ========================================================
    def _load_state(self) -> None:
        if not self.enable_resume:
            logger.info("Checkpoint resume disabled — starting from round 0")
            self.round = 0
            return

        if not self.state_path.exists():
            return

        try:
            data = json.loads(self.state_path.read_text())
            self.round = data.get("round", 0)
            logger.info(f"Resumed orchestrator from round {self.round}")
        except Exception as e:
            logger.warning(f"Failed to load state.json: {e}")


    def _save_state(self) -> None:
        if not self.state_path:
            return

        state = {
            "round": self.round,
            "workers": {wid: st.value for wid, st in self.workers.items()},
            "results_received": list(self.results.keys()),
        }

        self.state_path.write_text(json.dumps(state, indent=2))

    # ========================================================
    # WORKER CONNECTION PHASE
    # ========================================================
    async def on_worker_connected(self, worker_id: str) -> None:
        """
        Called immediately after a worker completes handshake.
        """
        self.workers[worker_id] = WorkerState.CONNECTED
        self.master_state = MasterState.WAITING_FOR_WORKERS
        self._save_state()

        logger.info(f"Worker {worker_id} connected")


    # ========================================================
    # RESULT COLLECTION
    # ========================================================
    async def receive_result(
        self,
        worker_id: str,
        data: bytes,
        round_no: int | None
    ) -> None:
        """
        Receive secure result from worker.
        """

        # ------------------------------
        # ROUND VALIDATION
        # ------------------------------
        if round_no is None:
            logger.warning(f"Missing round number from {worker_id}, ignoring")
            return

        if round_no != self.round:
            logger.warning(
                f"Stale/future result from {worker_id}: "
                f"got round {round_no}, expected {self.round}"
            )
            return

        # ------------------------------
        # WORKER VALIDATION
        # ------------------------------
        if worker_id not in self.workers:
            logger.warning(f"Result from unknown worker {worker_id}, ignoring")
            return

        if self.workers[worker_id] == WorkerState.FAILED:
            logger.warning(f"Result from failed worker {worker_id}, ignoring")
            return

        # ------------------------------
        # DUPLICATE PROTECTION
        # ------------------------------
        if worker_id in self.results:
            logger.warning(f"Duplicate result from {worker_id}, ignoring")
            return

        # ------------------------------
        # ACCEPT RESULT
        # ------------------------------
        self.results[worker_id] = data
        self.workers[worker_id] = WorkerState.WAITING_FOR_AGGREGATE
        self._save_state()

        logger.info(f"Result accepted from worker {worker_id}")


    def all_results_received(self) -> bool:
        """
        Check if round can be finalized.
        """
        active_workers = {
            wid: st for wid, st in self.workers.items()
            if st != WorkerState.FAILED
        }

        MIN_WORKERS = 1 # change later if needed

        # Not enough workers → do NOT finalize
        if len(active_workers) < MIN_WORKERS:
            logger.warning(
                f"Not enough active workers "
                f"({len(active_workers)}/{MIN_WORKERS})"
            )
            return False

        # All active workers must have sent results
        return all(
            st == WorkerState.WAITING_FOR_AGGREGATE
            for st in active_workers.values()
        )
    

    
    # ========================================================
    # EXPLICIT TRAINING CONTROL (CORE FIX)
    # ========================================================
    async def start_training(self, rounds: int, min_workers: int) -> None:
        # --- wait for READY workers ---
        while True:
            ready_workers = [
                wid for wid, st in self.workers.items()
                if st == WorkerState.READY
            ]
            if len(ready_workers) >= min_workers:
                break
            logger.info(f"Waiting for READY workers... {len(ready_workers)}/{min_workers}")
            await asyncio.sleep(0.5)

        self.training_active = True
        self.master_state = MasterState.ROUND_ACTIVE
        logger.info(f"Training started with {len(ready_workers)} workers")

        for r in range(rounds):
            self.round = r
            self.results.clear()

            await self.start_round(r)

            # BLOCK until all workers respond
            while not self.all_results_received():
                await asyncio.sleep(0.2)

            self.master_state = MasterState.AGGREGATING
            final_output = self.aggregate_results()
            if final_output is None:
                logger.warning(f"[ROUND {r}] No results to aggregate, skipping broadcast")
                continue  # skip broadcasting this round
            self.master_state = MasterState.BROADCASTING
            await self.broadcast_aggregated_model(final_output)

        self.training_active = False
        self.master_state = MasterState.IDLE
        logger.info("Training complete")

    async def start_round(self, round_no: int) -> None:
        logger.info(f"[MASTER] Starting round {round_no}")
        self.master_state = MasterState.ROUND_ACTIVE
        self.round = round_no
        self.results.clear()

        logger.info(f"[MASTER] Current workers: {list(self.workers.items())}")
        
        for worker_id, state in self.workers.items():
            logger.info(f"[MASTER] Worker {worker_id}: state={state}")
            if state == WorkerState.READY:
                logger.info(f"[MASTER] Worker {worker_id} is READY, sending model...")
                # Read model bytes
                with open(self.model_path, "rb") as f:
                    model_bytes = f.read()

                # Wrap in JSON
                payload = {
                    "type": "MODEL_FILE",
                    "payload_hex": model_bytes.hex()
                }
                await self.server.send_to_worker(worker_id, json.dumps(payload).encode())
                logger.info(f"Model sent to {worker_id} for round {round_no}")

                # Mark worker as TRAINING
                self.workers[worker_id] = WorkerState.TRAINING
            else:
                logger.info(f"[MASTER] Worker {worker_id} NOT READY (state={state}), skipping")

        self._save_state()

    def prepare_task_for(self, worker_id: str, round_no: int) -> bytes:
        """
        Send the current global model to the worker for training.
        """

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError("Global model file not found")

        # Send raw npz bytes
        with open(self.model_path, "rb") as f:
            model_bytes = f.read()

        logger.info(f"Prepared model for worker {worker_id} (round {round_no})")
        return model_bytes
    
    # ========================================================
    # AGGREGATION
    # ========================================================
    def aggregate_results(self):
        if not self.results:
            logger.warning("No results yet, waiting for workers...")
            return None  # don’t crash

        final_output = aggregate(self.results)
        return final_output
    
    # ========================================================
    # ROUND FINALIZATION
    # ========================================================
    def finalize_round(self, final_output: bytes) -> None:
        if not self.state_path:
            logger.warning("No state_path set; skipping output write")
            return

        out_dir = self.state_path.parent
        out_path = out_dir / f"aggregated_round_{self.round}.npz"

        with open(out_path, "wb") as f:
            f.write(final_output)

        logger.info(f"Aggregated model written to {out_path}")

        self.results.clear()
        self._save_state()

    async def on_worker_ready(self, worker_id: str, msg: bytes) -> None:
        logger.info(f"Worker {worker_id} sent READY")

        if worker_id not in self.workers:
            logger.warning(f"READY from unknown worker {worker_id}")
            return

        # Ignore READY if already training
        if self.workers[worker_id] == WorkerState.TRAINING:
            logger.info(f"Ignoring READY from {worker_id}, already TRAINING")
            return

        # Valid READY transitions
        if self.workers[worker_id] not in (
            WorkerState.CONNECTED,
            WorkerState.WAITING_FOR_AGGREGATE,
            WorkerState.WAITING_FOR_ROUND,
        ):
            logger.warning(
                f"READY from worker {worker_id} in invalid state {self.workers[worker_id]}"
            )
            return

        # JUST mark READY — nothing else
        self.workers[worker_id] = WorkerState.READY
        self._save_state()

        ready = sum(
            1 for st in self.workers.values()
            if st == WorkerState.READY
        )
        logger.info(f"[MASTER] READY workers = {ready}")

    async def broadcast_aggregated_model(self, final_output: bytes) -> None:
        logger.info("Broadcasting aggregated model to workers")

        for worker_id, state in self.workers.items():
            logger.info(f"[MASTER] Broadcasting to {worker_id}, current state: {state}")
            # Workers should be in WAITING_FOR_AGGREGATE state after sending results
            if state == WorkerState.WAITING_FOR_AGGREGATE:
                logger.info(f"[MASTER] Sending aggregated model ({len(final_output)} bytes) to {worker_id}")
                # Send aggregated model as protected message
                await self.server.send_to_worker(worker_id, final_output, msg_type="AGGREGATED_MODEL")
                logger.info(f"Aggregated model sent to {worker_id}")
                # Mark worker as ready for next round
                self.workers[worker_id] = WorkerState.READY
            else:
                logger.warning(f"[MASTER] Cannot broadcast to {worker_id}: not in WAITING_FOR_AGGREGATE state (state={state})")
        
        self.finalize_round(final_output)

        self.finalize_round(final_output)

    # ========================================================
    # MAIN EVENT LOOP
    # ========================================================
    async def run(self, port: int) -> None:
        """
        Event-driven orchestrator loop.
        """
        logger.info("Orchestrator starting (event-driven)")

        # Hook orchestrator callbacks to the server
        self.server.on_worker_ready = self.on_worker_ready  # implement this

        # Start the server (runs forever)
        await self.server.serve_forever(port)

