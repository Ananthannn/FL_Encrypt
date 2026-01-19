# Kimura

> **Post-Quantum Secure, Centralized File Transfer Protocol**

A fully centralized file transfer protocol designed to remain secure against classical and quantum adversaries. The protocol combines post-quantum cryptography for authentication and key exchange with high-performance symmetric encryption for data transfer.

---

## 1. Overview

This project implements a **Post-Quantum Secure Centralized File Transfer Protocol** that enables two peers to securely exchange files over an untrusted network.

Unlike traditional TLS-based systems, this protocol is designed from the ground up using **NIST-selected post-quantum primitives**, making it resistant to both classical and quantum attacks.

The system is **protocol-driven**, not just a secure application: message formats, cryptographic bindings, handshake phases, and failure behavior are explicitly defined.

---

## 2. Design Goals

* Post-quantum secure key exchange
* Post-quantum mutual authentication
* Forward secrecy
* Replay protection
* Fully Centralized
* Transport-agnostic (currently TCP)
* Modular cryptographic backend
* High-performance encrypted file transfer

---

## 3. Threat Model

### Adversary Capabilities

* Full network visibility (MITM)
* Packet injection and replay
* Passive traffic analysis
* Future quantum adversary

### Security Guarantees

* Confidentiality of transferred files
* Integrity and authenticity of peers
* Session key forward secrecy
* Resistance to replay and downgrade attacks

---

## 4. Cryptographic Primitives

| Purpose              | Algorithm   |
| -------------------- | ----------- |
| Key Exchange         | ML-KEM-768  |
| Authentication       | ML-DSA-65   |
| Symmetric Encryption | AES-256-GCM |
| Key Derivation       | HKDF-SHA256 |
| Hashing              | SHA-256     |

All cryptographic components are isolated into dedicated modules to allow future algorithm agility.

---

## 5. Protocol Roles

Although the system is centralized, each session defines temporary roles:

* **Initiator (I)** – Peer that starts the connection
* **Responder (R)** – Peer that accepts the connection

Roles are symmetric and exist only for the duration of the session.

---

## 6. Protocol Flow

```
Initiator                          Responder
   |--------- HELLO --------------------->|
   |<------ HELLO_ACK --------------------|
   |----- PQC_HANDSHAKE ----------------->|
   |<-- PQC_HANDSHAKE_RESP ---------------|
   |===== SECURE CHANNEL ESTABLISHED =====|
   |------ FILE_META -------------------->|
   |<----- META_ACK ----------------------|
   |------ FILE_CHUNKS ------------------>|
   |<----- TRANSFER_OK -------------------|
```

---

## 7. Handshake Specification

### 7.1 HELLO

Used for protocol negotiation and replay protection.

Fields:

* protocol_version
* peer_id
* supported_algorithms
* nonce_A

---

### 7.2 HELLO_ACK

Responder acknowledgement.

Fields:

* chosen_algorithms
* nonce_B

---

### 7.3 PQC_HANDSHAKE

Initiator authentication and key exchange.

Fields:

* ephemeral_kem_public_key
* long_term_dsa_public_key
* signature over (KEM key || nonce_A || nonce_B)

---

### 7.4 PQC_HANDSHAKE_RESP

Responder authentication and key agreement.

Fields:

* KEM ciphertext
* responder DSA public key
* signature over (ciphertext || nonce_A || nonce_B)

Both peers derive a shared session key using HKDF.

---

## 8. Secure Channel

After handshake completion, all messages are encrypted using AES-256-GCM.

Each encrypted frame contains:

* sequence number
* ciphertext
* authentication tag

Sequence numbers provide replay protection.

---

## 9. File Transfer Layer

### 9.1 FILE_META

Sent before data transfer.

Fields:

* filename
* filesize
* chunk_size
* file_hash (SHA-256)

---

### 9.2 FILE_CHUNK

Encrypted chunked file data.

Fields:

* chunk_index
* encrypted_payload

---

### 9.3 TRANSFER_OK

Indicates successful file reception and verification.

Fields:

* file_hash

---

## 10. Failure Handling

* Invalid signature → connection termination
* Replay detected → connection termination
* Decryption failure → connection termination
* Hash mismatch → file rejected

The protocol follows a **fail-closed** security model.

---

## 11. Implementation Details

* Written in Python (research based)
* CMake-based build system
* Modular cryptographic backend
* Deterministic test vectors
* Benchmark suite for handshake and throughput

---

## 12 Prerequisites (Ubuntu)

1. **liboqs shared library** (required for ML-KEM/ML-DSA):
   ```bash
   sudo apt install cmake ninja-build gcc g++ libssl-dev
   git clone https://github.com/open-quantum-safe/liboqs.git
   cd liboqs && mkdir build && cd build
   cmake -GNinja -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
   ninja && sudo ninja install && sudo ldconfig
   ```

2. **Python bindings**:
   ```bash
   pip install oqspy cryptography
   ```

## Verification Steps

Test the crypto backend before running peers:

```python
# test_crypto.py (add to tests/)
import oqs
kem = oqs.KeyEncapsulation('ML-KEM-768')
public_key = kem.generate_keypair()
print("PQC crypto ready:", len(public_key))
```

---

## 13. Folder Structure

```
pqc_secure_file_transfer/
├── benchmarks/
│   ├── handshake.py
│   ├── __init__.py
│   └── throughput.py
├── cli.py
├── crypto/
│   ├── aead.py
│   ├── hash.py
│   ├── __init__.py
│   ├── kdf.py
│   ├── keygen.py
│   ├── mldsa.py
│   ├── mlkem.py
│   └── signing.py
├── examples/
│   ├── __init__.py
│   ├── receive_file.py
│   └── send_file.py
├── file_transfer/
│   ├── bytes_conversion.py
│   ├── chunking.py
│   ├── __init__.py
│   └── transfer.py
├── flow.md
├── LICENSE
├── new_file.bin
├── protocol/
│   ├── constants.py
│   ├── __init__.py
│   ├── messages.py
│   └── state_machine.py
├── pyproject.toml
├── README.md
├── session/
│   ├── client.py
│   ├── __init__.py
│   ├── manager.py
│   └── server.py
├── tests/
│   ├── __init__.py
│   ├── test_file_integrity.py
│   ├── test_handshake.py
│   └── test_replay.py
└── transport/
    ├── __init__.py
    └── tcp.py
```

---

## 14. Project Status

* [x] Protocol specification
* [x] PQC handshake implementation
* [x] Secure file transfer
* [x] Benchmarks
* [ ] NAT traversal
* [ ] QUIC transport
* [ ] Formal security analysis

---

## 15. License

MIT License

---

## 16. Author

Project Owner and Protocol Designer

---

This project is intended as a **research-grade, protocol-focused implementation** demonstrating post-quantum secure P2P communication.
