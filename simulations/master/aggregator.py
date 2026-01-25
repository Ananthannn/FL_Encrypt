# simulations/master/aggregator.py

import io
import numpy as np
from typing import Dict

def aggregate(results: Dict[str, bytes]) -> bytes:
    """
    Dummy FedAvg-style aggregator.
    Input:
        results: {worker_id: npz_bytes}
    Output:
        aggregated npz bytes
    """

    updates = []

    for worker_id, raw_bytes in results.items():
        buf = io.BytesIO(raw_bytes)
        arr = np.load(buf, allow_pickle=True)

        updates.append({k: arr[k] for k in arr.files})

    if not updates:
        raise RuntimeError("No updates to aggregate")

    keys = updates[0].keys()

    avg_update = {
        k: sum(u[k] for u in updates) / len(updates)
        for k in keys
    }

    out = io.BytesIO()
    np.savez(out, **avg_update)
    out.seek(0)

    return out.read()
# ============================================================
# AGGREGATOR MODULE
# ============================================================