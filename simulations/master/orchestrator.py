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
from flwr.common import FitRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters
import io
import pickle
import lz4.frame
import torch
from kimura.protocol.fl_protocol import FLMessageType, serialize_fl_message
import numpy as np
logger = logging.getLogger("orchestrator")
from simulations.master.aggregator import create_med_strat

from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesRes, Status, Code
class DummyClientProxy(ClientProxy):
    def __init__(self, cid):
        super().__init__(cid)

    def get_parameters(self, ins, timeout): raise NotImplementedError()
    def fit(self, ins, timeout): raise NotImplementedError()
    def evaluate(self, ins, timeout): raise NotImplementedError()
    def reconnect(self, ins, timeout): raise NotImplementedError()
    def get_properties(self, ins, timeout):
        # Return empty properties since you don't use them
        return GetPropertiesRes(
            status=Status(code=Code.OK, message=""),
            properties={}
        )

# ============================================================
# ORCHESTRATOR CORE
# ============================================================
class Orchestrator:
    def __init__(
        self,
        server,
        state_path: Path | None = None,
        initial_model_path: Path | None = None,
        global_model: dict[str, torch.Tensor] | None = None,
        global_model_path: Path | None = None,
    ):
        """
        Args:
            server: SecureServer instance (crypto + TCP already inside)
            state_path: where master state.json is stored
            initial_model_path: path to initial PyTorch model
            global_model: in-memory model dict to use instead of loading from disk
            global_model_path: optional path to save global model checkpoints
        """
        self.server = server
        self.workers: Dict[str, WorkerState] = {}
        self.results: Dict[str, bytes] = {}
        self.round: int = 0
        self.initial_model_path = initial_model_path
        self.global_model_path = global_model_path or initial_model_path
        self.global_model = global_model
        self.training_active = False
        self.enable_resume = False
        self.master_state = MasterState.IDLE
        self.strategy = create_med_strat()
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
        # ACCEPT RESULT (FIXED)
        # ------------------------------
        try:
            # decompress and load the delta weights
            decompressed = lz4.frame.decompress(data)
            delta_weights = torch.load(io.BytesIO(decompressed), map_location="cpu")
            logger.info(f"Received delta keys from {worker_id}: {list(delta_weights.keys())}")
        except Exception as e:
            logger.error(f"Failed to deserialize delta from {worker_id}: {e}")
            self.workers[worker_id] = WorkerState.FAILED
            return

        # only mark as WAITING_FOR_AGGREGATE after successful load
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
        if self.global_model is None:
            if self.initial_model_path is None or not self.initial_model_path.exists():
                raise FileNotFoundError("Initial model not found")
            logger.info(f"[MASTER] Loading initial global model from {self.initial_model_path}")
            self.global_model = torch.load(self.initial_model_path, map_location="cpu",weights_only=True)
        logger.info(f"[MASTER] Starting round {round_no}")
        self.master_state = MasterState.ROUND_ACTIVE
        self.round = round_no
        self.results.clear()
        logger.info(f"[MASTER] Current workers: {list(self.workers.items())}")
        for worker_id, state in list(self.workers.items()):
            logger.info(f"[MASTER] Worker {worker_id}: state={state}")
            if state == WorkerState.READY:
                logger.info(f"[MASTER] Worker {worker_id} is READY, sending model...")
                # Serialize in-memory model
                state_dict_half = {
                    k: v.half() 
                    for k, v in (self.global_model.module.items() if isinstance(self.global_model, torch.nn.DataParallel) else self.global_model.items())
                }
                buffer = io.BytesIO()
                torch.save(state_dict_half, buffer)
                buffer.seek(0)
                model_bytes = buffer.getvalue()
                model_bytes = lz4.frame.compress(model_bytes)  # compressed for network
                size_mb = len(model_bytes) / 1024**2
                logger.info(f"[MASTER] Sending MODEL_FILE to {worker_id}, size={size_mb:.3f} MB")
                # Wrap for Flower protocol
                fl_bytes = serialize_fl_message(FLMessageType.MODEL_FILE, model_bytes)
                await self.server.send_to_worker(worker_id, FLMessageType.MODEL_FILE, fl_bytes)
                logger.info(f"Sent global model to {worker_id} for round {round_no}")
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
            return None
        flower_results = []
        for worker_id, payload_bytes in self.results.items():
            if not isinstance(payload_bytes, (bytes, bytearray)):
                logger.error(f"Worker {worker_id} returned unexpected type {type(payload_bytes)}")
                continue
            try:
                # -------------------------
                # Decompress ΔW from worker
                # -------------------------
                decompressed = lz4.frame.decompress(payload_bytes)
                payload = torch.load(io.BytesIO(decompressed), map_location="cpu")
                # Extract delta and num_examples
                delta_weights = payload["delta"]
                num_examples = payload["num_examples"]
                # Convert to numpy arrays for Flower
                arrays = {k: v.numpy() for k, v in delta_weights.items()}
                parameters = ndarrays_to_parameters([arrays[k] for k in sorted(arrays.keys())])
            except Exception as e:
                logger.error(f"Failed to deserialize update from {worker_id}: {e}")
                continue
            proxy = DummyClientProxy(worker_id)
            fit_res = FitRes(
                status=Status(code=Code.OK, message=""),
                parameters=parameters,
                num_examples=num_examples,
                metrics={},
            )
            flower_results.append((proxy, fit_res))
        if not flower_results:
            logger.error("No valid worker updates received")
            return None
        # -------------------------
        # Aggregate via Flower strategy
        # -------------------------
        aggregated = self.strategy.aggregate_fit(
            server_round=self.round,
            results=flower_results,
            failures=[],
        )
        if aggregated is None:
            logger.warning("Aggregation returned None")
            return None
        aggregated_params, metrics = aggregated
        ndarrays = parameters_to_ndarrays(aggregated_params)
        # -------------------------
        # Update in-memory global model
        # -------------------------
        if self.global_model is None:
            if self.initial_model_path is None or not self.initial_model_path.exists():
                raise FileNotFoundError("No initial model available to send")
            logger.info(f"Loading initial global model from {self.initial_model_path}")
            self.global_model = torch.load(self.initial_model_path, map_location="cpu",weights_only=True)

        for k, v in zip(self.global_model.keys(), ndarrays):
            self.global_model[k] = torch.tensor(v)
        # -------------------------
        # Serialize and compress global model
        # -------------------------
        buffer = io.BytesIO()
        torch.save(self.global_model, buffer)  # just save the dict directly
        serialized_bytes = lz4.frame.compress(buffer.getvalue())
        logger.info(f"Aggregation complete: {metrics}")
        return serialized_bytes

            
    # ========================================================
    # ROUND FINALIZATION
    # ========================================================
    def finalize_round(self, final_output: bytes) -> None:
        if not self.state_path:
            logger.warning("No state_path set; skipping output write")
            return

        # Skip writing the .npz completely
        # out_dir = self.state_path.parent
        # out_path = out_dir / f"aggregated_round_{self.round}.npz"
        # with open(out_path, "wb") as f:
        #     f.write(final_output)
        # logger.info(f"Aggregated model written to {out_path}")

        self.results.clear()
        self._save_state()  # you can keep this if you still want internal state persistence

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
            if state == WorkerState.WAITING_FOR_AGGREGATE:
                logger.info(f"[MASTER] Sending aggregated model ({len(final_output)} bytes) to {worker_id}")
                await self.server.send_to_worker(
                    worker_id,
                    FLMessageType.AGGREGATED_MODEL,
                    final_output
                )
                logger.info(f"Aggregated model sent to {worker_id}")
                self.workers[worker_id] = WorkerState.READY
            else:
                logger.warning(f"[MASTER] Cannot broadcast to {worker_id}: not in WAITING_FOR_AGGREGATE state (state={state})")
        # Save aggregated model as new global model
        if self.global_model_path:
            self.global_model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.global_model_path, "wb") as f:
                f.write(final_output)
            logger.info(f"Global model updated at {self.global_model_path}")
        self.finalize_round(final_output)

    # ========================================================
    # MAIN EVENT LOOP
    # ========================================================
    async def run(self, port: int) -> None:
        """
        Event-driven orchestrator loop.
        """
        logger.info("Orchestrator starting up...")

        # Hook orchestrator callbacks to the server
        self.server.on_worker_ready = self.on_worker_ready  # implement this
        # Start the server (runs forever)
        await self.server.serve_forever(port)