Crypto & protocol primitives → functions

State, sessions, peers → classes

---

# 🔐 Cryptographic Rules for Implementing Your Protocol

These are **non-negotiable** rules when working with PQC + AEAD.

---

## RULE 1 — **Never implement crypto primitives yourself**

❌ Don’t implement:

* AES
* GCM
* ML-KEM
* ML-DSA
* Hash functions

✅ Always call:

* **liboqs** for PQC
* **cryptography** for AEAD / KDF / hashes

**Reason:** side-channels, constant-time, padding, randomness — impossible to get right.

---

## RULE 2 — **Use authenticated encryption only**

❌ Never use:

* AES-CBC
* AES-CTR
* Encrypt-then-MAC DIY schemes

✅ Use:

* **AES-256-GCM only**

**Reason:** encryption without authentication = broken protocol.

---

## RULE 3 — **Keys have single, narrow purposes**

One key → **one job**.

| Key                  | Purpose                |
| -------------------- | ---------------------- |
| ML-KEM shared secret | Key agreement only     |
| HKDF output          | Session key derivation |
| AES key              | Data encryption only   |
| ML-DSA key           | Authentication only    |

❌ Never reuse keys across roles.

---

## RULE 4 — **Derive session keys properly (mandatory)**

❌ Wrong:

```python
aes_key = shared_secret[:32]
```

✅ Correct:

```python
aes_key = HKDF(
    algorithm=SHA256(),
    length=32,
    salt=handshake_hash,
    info=b"pqc-p2p-session"
).derive(shared_secret)
```

**Reason:** raw KEM output is not uniformly safe as an AEAD key.

---

## RULE 5 — **Every session uses fresh PQC material**

For **every connection**:

* New ML-KEM keypair
* New nonces
* New session key

❌ Never reuse KEM keys across sessions.

This gives **forward secrecy**.

---

## RULE 6 — **All cryptographic context must be bound**

Signatures must cover:

* Nonces
* Algorithm choices
* KEM public keys or ciphertexts

Example:

```text
Sign(
  kem_pk || nonce_A || nonce_B || protocol_version
)
```

**Reason:** prevents downgrade and MITM attacks.

---

## RULE 7 — **Nonces are NEVER reused (AEAD law)**

For AES-GCM:

* Nonce must be **unique per key**
* 96-bit nonces recommended

Best practice:

```python
nonce = session_nonce_prefix || sequence_number
```

❌ Never randomize blindly without tracking.

Nonce reuse = **catastrophic failure**.

---

## RULE 8 — **Sequence numbers are mandatory**

Every encrypted message must include:

* Monotonic sequence number
* Checked on receive

❌ No sequence number → replay attacks.

---

## RULE 9 — **Fail closed, always**

If **anything** fails:

* Signature verification
* Decryption
* Replay check
* Hash mismatch

👉 **Terminate the connection immediately**

❌ Never attempt recovery.

---

## RULE 10 — **Crypto errors are silent**

❌ Never leak:

* Which check failed
* Partial state
* Timing differences

Good:

```python
raise ProtocolAbort()
```

Bad:

```text
Invalid signature from peer
```

**Reason:** side-channel & oracle prevention.

---

## RULE 11 — **No plaintext metadata after handshake**

After secure channel:

* Even file metadata is encrypted
* No filenames or sizes in cleartext

---

## RULE 12 — **Replay protection is explicit**

Use:

* Nonces in handshake
* Sequence numbers in data
* Store highest seen sequence

---

## RULE 13 — **Trust model must be explicit**

You must choose one:

* 🔒 Pre-shared public keys
* 🔐 Trust-on-First-Use (TOFU)
* 🧾 Certificate-based (rare in P2P)

And document it.

Undocumented trust = broken protocol.

---

## RULE 14 — **Benchmark crypto, not Python**

Measure:

* KEM time
* Signature time
* AES-GCM throughput

❌ Don’t benchmark:

* Serialization
* Logging
* Print statements

---

## RULE 15 — **Version everything**

Every message includes:

* Protocol version
* Algorithm IDs

❌ No implicit defaults.

---

## RULE 16 — **Keep crypto code boring**

If your crypto code looks:

* clever
* short
* fancy

It’s probably wrong.

Correct crypto code is **boring, explicit, and repetitive**.

---

## RULE 17 — **Document every crypto decision**

Every crypto choice must answer:

* Why this algorithm?
* Why this key size?
* Why this mode?

This is what reviewers look for.

---

# Minimal crypto checklist (print this)

Before you say “done”:

* [ ] liboqs only for PQC
* [ ] cryptography only for AEAD/KDF
* [ ] HKDF used for all session keys
* [ ] Fresh KEM per session
* [ ] AES-GCM with unique nonces
* [ ] Sequence numbers enforced
* [ ] Fail-closed on all errors
* [ ] No crypto reimplementation
* [ ] Trust model documented

---
Always pass bytes, not string or int, to crypto primitives.


Server shows:

```
$ pqc-session server --port 8443 --dataset cifar10 --key ~/.pqc/server_key
[2026-01-11 14:54] Listening on 0.0.0.0:8443
[2026-01-11 14:55] Client gpu-01 connected (PQC handshake OK)
[2026-01-11 14:55] Client gpu-02 connected (PQC handshake OK)
[2026-01-11 14:55] ✓ All 10 clients connected. Channel established.
[2026-01-11 14:55] ✓ Ready for federated training transfer.
Round 1/100: Broadcasting CIFAR10 model (2.1MB)...
```
Each Client shows:
```
$ pqc-session client trainer:8443 --gpu 0 --key ~/.pqc/client_key
[2026-01-11 14:55] Connecting to trainer.example.com:8443
[2026-01-11 14:55] ✓ PQC Handshake complete (ML-KEM-1024)
[2026-01-11 14:55] ✓ Secure channel established.
[2026-01-11 14:55] ✓ Ready for training transfer.
Round 1/100: Received model, training on GPU 0...
```
---
```
fallocate -l 1G new_file.bin
```