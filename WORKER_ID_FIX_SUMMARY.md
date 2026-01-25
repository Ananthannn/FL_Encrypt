# Worker ID Fix: Connection Counter → Cryptographic Identity

## Problem
- **Old Behavior**: Workers were identified by connection order (0, 1, 2, 3...) using `clients_processed` counter
- **Issue**: Gaps in sequence (1, 3, 5, 7) appeared when some workers failed to connect
- **Root Cause**: Connection index ≠ worker identity. Two different concepts were conflated.

## Solution
Worker IDs are now **logically derived from cryptographic identity** during TLS/PQC handshake:

```
worker_id = SHA256(peer_public_key)[:16]  # 16 hex chars = 64 bits
```

This ensures:
✅ **Stable**: Same worker always gets same ID regardless of connection order  
✅ **Unique**: ID is cryptographically bound to worker's signing key  
✅ **Deterministic**: Reproducible across restarts  
✅ **No Gaps**: 1, 3, 5, 7 problem disappears (now called `a1b2c3d4e5f67890`, `c7d8e9f0a1b2c3d4`, etc.)

---

## Changed Files

### 1. [kimura/session/manager.py](kimura/session/manager.py)
**What Changed:**
- Added `self.worker_id` field to `SessionManager.__init__()` (line 24)
- Extract worker_id during server-side handshake and store in SessionManager (line 44-45)

**Code:**
```python
# __init__ now has:
self.worker_id = None  # Will be set during handshake

# establish_channel() now extracts and stores:
peer_pubkey = self.state_machine.get_peer_identity_key()
worker_id = hashlib.sha256(peer_pubkey).hexdigest()[:16]
self.worker_id = worker_id  # ← NEW: Store in SessionManager
```

### 2. [kimura/session/master.py](kimura/session/master.py)
**What Changed:**
- Removed `clients_processed` counter (line 32)
- `handle_client()` now extracts `worker_id` from SessionManager instead of using counter
- All client_id references renamed to worker_id for clarity

**Code:**
```python
# OLD:
self.clients_processed = 0
client_id = self.clients_processed
self.clients_processed += 1

# NEW:
worker_id = mgr.worker_id  # From handshake crypto
self.active_clients[worker_id] = (reader, writer, mgr)
```

### 3. [simulations/master/run_master.py](simulations/master/run_master.py)
**What Changed:**
- Removed redundant `str(worker_id)` conversions (already a string from handshake)
- Updated log messages for consistency

**Code:**
```python
# OLD:
worker_id = str(worker_id)  # unnecessary

# NEW:
# worker_id is already a string from cryptographic hash
```

---

## Format of New Worker IDs

**Before:**
```json
{
  "workers": {
    "1": "RESULT_RECEIVED",
    "3": "RESULT_RECEIVED",
    "5": "RESULT_RECEIVED"
  }
}
```

**After (Example):**
```json
{
  "workers": {
    "a7f2b8c9d1e3f5g6": "RESULT_RECEIVED",
    "c4e6f8a0b2d4e6f8": "RESULT_RECEIVED",
    "e9g1a3b5c7d9e1f3": "RESULT_RECEIVED"
  }
}
```

Each worker ID is **unique per worker** and **stable across runs**.

---

## Flow Summary

```
Worker Startup
    ↓
    Connect to Master (TCP)
    ↓
    Exchange PQC Handshake (MLKEM + MLDSA)
    ↓
    Master extracts peer_pubkey from signature
    ↓
    Master calculates: worker_id = SHA256(peer_pubkey)[:16]
    ↓
    SessionManager stores in self.worker_id
    ↓
    Master uses worker_id as identity (not connection counter)
    ↓
    master_state.json persists worker_id keys
```

---

## Backward Compatibility

⚠️ **Breaking Change**: `master_state.json` keys changed from `"0", "1", "2"` to `"a7f2b8c9d1e3f5g6"`, etc.

If you have existing state files, they will not be compatible with the old format.

---

## Testing

To verify the fix:

```bash
# 1. Start master
cd simulations
python3 master/run_master.py

# 2. In another terminal, start workers (in any order)
python3 workers/run_worker.py  # Will get ID like "a7f2b8c9d1e3f5g6"
python3 workers/run_worker.py  # Will get different ID like "c4e6f8a0b2d4e6f8"

# 3. Check master_state.json
cat simulations/shared/master_state.json

# 4. Observe:
# - NO MORE CONNECTION COUNTER ORDER (0, 1, 2...)
# - Worker IDs are stable cryptographic hashes
# - No gaps (previous: 1, 3, 5, 7)
```

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Identity | Connection order | Cryptographic hash |
| Stable across restarts | ❌ No (rebuilds sequentially) | ✅ Yes (same pubkey = same ID) |
| No gaps in sequences | ❌ (if some fail to connect) | ✅ (ID independent of timing) |
| Uniqueness | ❌ (only per session) | ✅ (lifetime of key) |
| Semantic clarity | ❌ (counter ≠ identity) | ✅ (hash = identity) |

