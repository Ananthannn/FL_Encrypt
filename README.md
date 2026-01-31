# Post-Quantum Secure Distributed CNN Training

## What This Project Does

Implements a secure distributed and federated training framework for deep Convolutional Neural Networks (CNNs) using post-quantum digital signatures.

The system digitally signs model updates during training and aggregation to ensure:
- Model authenticity
- Integrity of exchanged parameters
- Protection against tampering, impersonation, and replay attacks

The framework is designed to remain secure against future quantum adversaries and is evaluated in both centralized and decentralized federated learning setups.

## Tools & Technologies Used

- **Python**
- **Deep Learning Frameworks**
  - TensorFlow / PyTorch (CNN training)
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
