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
from enum import Enum
from pathlib import Path
from typing import Dict, Any
from shared.state import MasterState, WorkerState
logger = logging.getLogger("orchestrator")

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
        self.initial_model_sent: set[str] = set()

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
        ready_workers = [
            wid for wid, st in self.workers.items()
            if st == WorkerState.READY
        ]

        if len(ready_workers) < min_workers:
            raise RuntimeError("Not enough READY workers")

        self.training_active = True
        self.master_state = MasterState.ROUND_ACTIVE

        logger.info(f"Training started with {len(ready_workers)} workers")

        for r in range(rounds):
            self.round = r
            self.results.clear()

            await self.start_round(r)

            # 🔒 BLOCK until all workers respond
            while not self.all_results_received():
                await asyncio.sleep(0.2)

            self.master_state = MasterState.AGGREGATING
            final_output = self.aggregate_results()

            self.master_state = MasterState.BROADCASTING
            await self.broadcast_aggregated_model(final_output)

            # Workers go back to WAITING_NEXT_ROUND
            #for wid in self.workers:
            #    if self.workers[wid] != WorkerState.FAILED:
            #        self.workers[wid] = WorkerState.WAITING_NEXT_ROUND

        self.training_active = False
        self.master_state = MasterState.IDLE
        logger.info("Training complete")

    async def start_round(self, round_no: int) -> None:
        logger.info(f"Starting round {round_no}")
        self.master_state = MasterState.ROUND_ACTIVE
        self.results.clear()

        for worker_id, state in self.workers.items():
            if state == WorkerState.READY:
                payload = self.prepare_task_for(worker_id, round_no)

                await self.server.send_to_worker(
                    worker_id,
                    payload,
                    msg_type="START_ROUND"
                )

                self.workers[worker_id] = WorkerState.TRAINING

        self._save_state()


    def prepare_task_for(self, worker_id: str, round_no: int) -> bytes:
        """
        Prepare task payload for a specific worker.
        This is a placeholder and should be customized.
        """
        task = {
            "round": round_no,
            "task_data": f"Task for worker {worker_id} in round {round_no}"
        }
        return json.dumps(task).encode('utf-8')
    
    # ========================================================
    # AGGREGATION
    # ========================================================
    def aggregate_results(self) -> Any:
        """
        Aggregate worker outputs using aggregator module.
        """
        logger.info("Aggregating worker results")

        from simulations.master.aggregator import aggregate

        final_output = aggregate(self.results)
        return final_output


    # ========================================================
    # ROUND FINALIZATION
    # ========================================================
    def finalize_round(self, final_output: Any) -> None:
        self.server.write_output(final_output)
        self.results.clear()
        logger.info(f"Round {self.round} finalized")
        self._save_state()

    async def on_worker_ready(self, worker_id: str, msg: bytes) -> None:
        logger.info(f"Worker {worker_id} sent READY")

        if worker_id not in self.workers:
            logger.warning(f"READY from unknown worker {worker_id}")
            return

        # Ignore READY if already training
        if self.workers.get(worker_id) == WorkerState.TRAINING:
            logger.info(f"Ignoring READY from worker {worker_id}, already TRAINING")
            return

        if self.workers[worker_id] not in (
            WorkerState.CONNECTED,
            WorkerState.WAITING_FOR_AGGREGATE,
            WorkerState.WAITING_FOR_ROUND
        ):
            logger.warning(
                f"READY from worker {worker_id} in state {self.workers[worker_id]}"
            )
            return

        # Send initial model ONLY once
        if worker_id not in self.initial_model_sent:
            self.initial_model_sent.add(worker_id)  # mark first!
            try:
                if not self.model_path or not self.model_path.exists():
                    raise FileNotFoundError(self.model_path)

                await self.server.send_file(worker_id, self.model_path)
                logger.info(f"Initial model sent to {worker_id}")

                # Set worker to TRAINING after sending
                self.workers[worker_id] = WorkerState.TRAINING
                self._save_state()

            except Exception as e:
                logger.error(f"Failed to send model to {worker_id}: {e}")
                self.workers[worker_id] = WorkerState.FAILED
                self._save_state()
                return
        else:
            # No need to send model again → safe to mark READY
            self.workers[worker_id] = WorkerState.READY

        # Update master state
        if all(
            st == WorkerState.READY
            for st in self.workers.values()
            if st != WorkerState.FAILED
        ):
            self.master_state = MasterState.IDLE

        self._save_state()
    async def broadcast_aggregated_model(self, final_output: Any) -> None:
        """ Broadcast aggregated model to all READY workers.
        """
        logger.info("Broadcasting aggregated model to workers")

        # Serialize final output (customize as needed)
        payload = json.dumps({
            "round": self.round,
            "model": final_output  # adjust serialization as needed
        }).encode('utf-8')

        for worker_id, state in self.workers.items():
            if state == WorkerState.READY:
                await self.server.send_to_worker(
                    worker_id,
                    payload,
                )
                logger.info(f"Aggregated model sent to {worker_id}")

        # Finalize round
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

