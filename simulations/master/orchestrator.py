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
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any
from shared.state import WorkerState
logger = logging.getLogger("orchestrator")

# ============================================================
# ORCHESTRATOR CORE
# ============================================================
class Orchestrator:
    def __init__(self, server, state_path: Path | None = None):
        """
        Args:
            server: SecureServer instance (crypto + TCP already inside)
            state_path: where master state.json is stored
        """
        self.server = server

        self.workers: Dict[str, WorkerState] = {}
        self.results: Dict[str, bytes] = {}

        self.round: int = 0

        self.state_path = state_path
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_state()

        logger.info("Orchestrator initialized")


    # ========================================================
    # STATE MANAGEMENT
    # ========================================================
    def _load_state(self) -> None:
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
        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerState.CONNECTED
            self._save_state()

        self.workers[worker_id] = WorkerState.HANDSHAKE_DONE
        self._save_state()

        logger.info(f"Worker {worker_id} connected and handshake complete")

    # ========================================================
    # TASK ORCHESTRATION
    # ========================================================
    async def dispatch_task(self, worker_id: str, payload: bytes) -> None:
        """
        Securely send task to worker.
        """
        try:
            await self.server.send_to_worker(worker_id, payload)
            self.workers[worker_id] = WorkerState.TASK_SENT
            self._save_state()

            logger.info(f"Task dispatched to worker {worker_id}")
        except Exception as e:
            self.workers[worker_id] = WorkerState.FAILED
            self._save_state()

            logger.error(f"Task dispatch failed for {worker_id}: {e}")


    async def dispatch_to_all(self, payload: bytes) -> None:
        """
        Send same task to all ready workers.
        """
        for worker_id, state in self.workers.items():
            if state == WorkerState.HANDSHAKE_DONE:
                await self.dispatch_task(worker_id, payload)


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
        self.workers[worker_id] = WorkerState.RESULT_RECEIVED
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

        MIN_WORKERS = 2  # change later if needed

        # Not enough workers → do NOT finalize
        if len(active_workers) < MIN_WORKERS:
            logger.warning(
                f"Not enough active workers "
                f"({len(active_workers)}/{MIN_WORKERS})"
            )
            return False

        # All active workers must have sent results
        return all(
            st == WorkerState.RESULT_RECEIVED
            for st in active_workers.values()
        )



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
        """
        Persist final output and advance round.
        """
        self.server.write_output(final_output)

        self.round += 1
        self.results.clear()

        logger.info(f"Round {self.round} finalized")
        self._save_state()

    async def on_worker_ready(self, worker_id: str, msg: bytes) -> None:
        logger.info(f"Worker {worker_id} ready with message: {msg}")
        payload = self.prepare_task_for(worker_id)  # implement this
        await self.server.send_to_client(worker_id, payload)
        self.workers[worker_id] = WorkerState.TRAINING
        self._save_state()

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

