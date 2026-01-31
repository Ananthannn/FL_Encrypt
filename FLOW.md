

## 1️⃣ Conceptual walkthrough of `orchestrator.py`

Think of `orchestrator.py` as the **central control brain** for the master side of your system. Here’s what it does:

### a) Initialization

* Loads **paths to keys** (ML-KEM, ML-DSA) from `simulations/keys`
* Initializes logging
* Reads configuration files (`configs/master.yaml`, `configs/experiment.yaml`) if needed
* Creates a **SecureServer instance**, which internally uses the `StateMachine` for:

  * Handshake
  * Session key derivation
  * AEAD context creation

### b) Accepting worker connections

* Listens for TCP connections on `DEFAULT_PORT`
* For each worker:

  * Accepts connection
  * Runs handshake using `StateMachine`
  * Stores the `AEADPair` context (send/recv keys & sequence numbers) for secure communication

### c) Task orchestration

* Determines **which files / model updates / data** to send to each worker
* Sends tasks securely (using `StateMachine.send_protected`)
* Receives results (model updates, metrics, etc.) securely (using `StateMachine.recv_protected`)

### d) Aggregation / Finalization

* After receiving results from all workers:

  * Calls **aggregator** functions (`aggregator.py`) to combine worker outputs
  * Writes final results to `simulations/shared` (like `received_from_workers.pt`)

### e) State management

* Maintains **per-worker states**, e.g.:

  * `INIT` → `HANDSHAKE_SENT` → `HANDSHAKE_COMPLETE` → `DATA_SENT` → `RESULT_RECEIVED`
* Updates the master’s view of **who is ready**, **who failed**, etc.

---

## 2️⃣ Dependencies (crypto + ML)

### Crypto dependencies:

* `kimura/crypto/mlkem.py` → ML-KEM encaps/decaps
* `kimura/crypto/mldsa.py` → ML-DSA signatures
* `crypto/signing.py` → load keys, sign, verify, TOFU checks
* `kimura/file_transfer/transfer.py` → `chunked_send_file`, `recv_file`, `send_length_prefixed`, `recv_length_prefixed`
* `kimura/protocol/state_machine.py` → handshake + AEAD key derivation

### ML / Federated learning dependencies:

* `model/server_model.py` → model aggregation on master side
* `simulations/master/aggregator.py` → aggregate worker updates
* `simulations/workers/trainer.py` → defines worker-side training; orchestrator only coordinates these
* `model/train.py` → optional if orchestrator runs experiments

### Other utilities:

* `kimura/protocol/constants.py` → constants like `DEFAULT_PORT`, `ML_DSA_65_SIG_LEN`, etc.
* `kimura/transport/tcp.py` → TCP transport abstraction

So basically, **orchestrator.py doesn’t do ML itself**, it **coordinates workers that train models** and uses crypto to secure all comms.

---

## 3️⃣ Simulations folder after execution

Current structure:

```
simulations/
 ├─ keys/
 ├─ logs/
 ├─ master/
 ├─ shared/
 └─ workers/
```

### After a run:

1. **keys/** → remains the same (client/server ML-KEM + ML-DSA keys)
2. **logs/** → populated with worker connections, handshake logs, AEAD key events, errors
3. **master/checkpoints/** → updated if you save intermediate states (`state.json`)
4. **shared/** → final aggregated outputs (e.g., `received_from_workers.pt`)
5. **workers/datasets/** → unchanged
6. **workers/state.json** → updated with last known worker state

✅ Typically, `simulations/` remains inside the project root (`FL_Encrypt/`) unless you explicitly move or archive it.

---

## 4️⃣ Core functions of `orchestrator.py`

Here’s what you’d usually see in `orchestrator.py`:

| **Function / Role**                     | **Purpose**                                     | **State Updates**                        |
| --------------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| `__init__()`                            | Initialize server, load keys, configure logging | Master state = INIT                      |
| `accept_worker()`                       | Accept TCP connections from workers             | Per-worker state = INIT → HANDSHAKE_SENT |
| `handshake_worker(worker_conn)`         | Run `StateMachine` handshake                    | Per-worker state = HANDSHAKE_COMPLETE    |
| `dispatch_task(worker_conn, task)`      | Send model/data to worker                       | State = DATA_SENT                        |
| `collect_result(worker_conn)`           | Receive results from worker                     | State = RESULT_RECEIVED                  |
| `aggregate_results()`                   | Call aggregator.py to merge updates             | Master state = RESULTS_AGGREGATED        |
| `error_handler(worker_conn, exception)` | Retry / log errors                              | State = ERROR                            |
| `shutdown()`                            | Close connections safely                        | Master state = SHUTDOWN                  |

---

### ✅ TL;DR

* **Orchestrator.py = brain**: coordinates all workers, secures comms, aggregates results
* **Depends on**:

  * Crypto: `kimura/crypto`, `crypto/signing`, `file_transfer`
  * ML: `model/server_model.py`, `aggregator.py`
* **Simulations folder**: remains inside project root; stores keys, logs, shared results, and checkpoints
* **State updates**: tracked **per-worker** and for master overall, from INIT → HANDSHAKE → DATA → RESULT → AGGREGATED

---

If you want, I can **draw a full visual pipeline diagram** of `orchestrator.py` + master + worker + crypto + ML so you can see **exactly how data and state flow**.

Do you want me to make that diagram?


simulations/
├── master/
│   ├── run_master.py        ← starts networking + waits
│   ├── orchestrator.py     ← FL brain (rounds, quorum, state)
│   ├── trainer.py          ← loads model, aggregation logic
│   └── start_training.py   ← HUMAN TRIGGER (IMPORTANT)
│
├── workers/
│   ├── run_worker.py       ← worker process
│   └── train.py            ← AI guy code (GPU)
│
└── shared/
    ├── model.npz
    ├── round_*.npz
    └── master_state.json
