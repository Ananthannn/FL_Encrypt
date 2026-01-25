"""2️⃣ Where training actually happens

Look at simulations/workers/train.py. This is probably your worker-local training code.

Each worker loads its local dataset.

Uses the callback function to:

Receive the server model.

Train on local data.

Send updated weights back to the master.

So the GPU / ML training is only in the worker-side train.py logic (or whatever you plug into weights_callback), not in worker.py (the networking module)."""
"""✅ Local dataset + GPU training"""