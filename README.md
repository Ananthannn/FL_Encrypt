# Post-Quantum Secure Distributed CNN Training

> **⚠️ DEVELOPMENT STATUS:** This project is currently under active development and is not ready for production use. Please do not use this framework at this stage.

## What This Project Does
Implements a secure distributed and federated training framework for deep Convolutional Neural Networks (CNNs) using post-quantum digital signatures.

The system digitally signs model updates during training and aggregation to ensure:
- Model authenticity
- Integrity of exchanged parameters
- Protection against tampering, impersonation, and replay attacks

The framework is designed to remain secure against future quantum adversaries and is evaluated in both centralized and decentralized federated learning setups.

## What is Federated Learning?

Federated learning is a machine learning approach that allows multiple parties to collaboratively train a shared model without sharing their raw data. Unlike traditional machine learning where all training data is collected in one central location, federated learning keeps data distributed across multiple devices or organizations.

**How it works:**
1. A central server initializes a global model and distributes it to multiple participants (clients)
2. Each client trains the model locally on their own private data
3. Clients send only their model updates (not the raw data) back to the server
4. The server aggregates these updates to improve the global model
5. This process repeats until the model converges

**Key benefits:**
- **Privacy preservation:** Sensitive data never leaves its source
- **Reduced bandwidth:** Only model updates are transmitted, not entire datasets
- **Decentralization:** Enables collaboration without centralizing data storage
- **Compliance:** Helps meet data protection regulations like GDPR

**Real-world applications:**
- Smartphones learning from user behavior without uploading personal data
- Hospitals collaborating on medical AI without sharing patient records
- Financial institutions detecting fraud while keeping transactions private

**Why this matters for this project:**
In this project, we add an extra layer of security to federated learning by using post-quantum cryptographic signatures to verify that model updates are authentic and haven't been tampered with during transmission. This protects against attackers who might try to poison the model or impersonate legitimate participants.

## Tools & Technologies Used

- **Python**
- **Deep Learning Frameworks**
  - TensorFlow (CNN training)
- **Federated Learning**
  - Custom distributed training pipeline
- **Post-Quantum Cryptography**
  - ML-KEM-768
  - ML-DSA-65
  - AES-GCM
- **Cryptographic Libraries**
  - NIST-standardized PQC implementations
- **GPU Acceleration**
  - CUDA-enabled training (where available)
- **Experiment Tracking**
  - Logging and performance evaluation tools
