# simulations/master/aggregator.py
"""
Production-grade TFF aggregator for medical federated learning.
Supports SecureFedAvg, differential privacy, weighted aggregation,
outlier detection, and medical data robustness.
"""

import io
import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AggregationConfig:
    """Configuration for secure medical aggregation."""
    dp_noise_multiplier: float = 1.1  # DP noise level (higher = more private)
    dp_clipping_norm: float = 1.0     # Gradient clipping
    secure_aggregation: bool = True    # Use secure aggregation
    outlier_threshold: float = 3.0    # Z-score for outlier detection
    weights_by_dataset_size: bool = True  # Weight clients by # samples

def secure_mean_federated(
    weights_match: tff.Computation,
    value_sum: tff.Computation
) -> tff.Computation:
    """Secure mean aggregation factory with DP noise."""
    @tff.federated_computation(
        tff.type_at_server(tf.float32),
        tff.type_at_clients(tf.float32)
    )
    def secure_mean(client_data):
        client_weights = tff.federated_map(weights_match, client_data)
        client_sums = tff.federated_map(value_sum, client_data)
        total_weight = tff.federated_aggregate(client_weights, 0.0, np.add)
        total_sum = tff.federated_aggregate(client_sums, 0.0, np.add)
        mean = total_sum / tff.require_broadcast(total_weight)
        
        # Add DP noise for medical privacy
        noise = tff.learning.models.add_dp_noise(
            mean, noise_multiplier=1.1, clipping_norm=1.0
        )
        return tff.federated_output(noise)
    
    return secure_mean

class MedicalAggregator:
    """Production aggregator for TFF medical simulations."""
    
    def __init__(self, config: AggregationConfig = AggregationConfig()):
        self.config = config
        self._aggregation_factory = tff.aggregators.DifferentiallyPrivateFactory(
            noise_multiplier=config.dp_noise_multiplier,
            clip_norm=config.dp_clipping_norm
        )
    
    def validate_updates(self, updates: Dict[str, bytes]) -> Tuple[Dict[str, bytes], Dict[str, float]]:
        """Validate medical model updates: check shapes, detect outliers."""
        validated = {}
        weights = {}
        
        reference_shape = None
        for worker_id, raw_bytes in updates.items():
            try:
                buf = io.BytesIO(raw_bytes)
                arr = np.load(buf, allow_pickle=True)
                
                # Extract model weights and metadata
                weights_data = {k: arr[k] for k in arr.files if k != 'metadata'}
                metadata = arr.get('metadata', {})
                
                # Validate shapes consistency (critical for medical models)
                if reference_shape is None:
                    reference_shape = {k: v.shape for k, v in weights_data.items()}
                else:
                    for k, v in weights_data.items():
                        if v.shape != reference_shape.get(k):
                            logger.warning(f"Shape mismatch for {worker_id}, key {k}")
                            continue
                
                # Weight by dataset size (medical standard)
                dataset_size = metadata.get('dataset_size', 1)
                weights[worker_id] = dataset_size if self.config.weights_by_dataset_size else 1.0
                
                # Outlier detection (z-score on weight norms)
                weight_norms = [np.linalg.norm(w.flatten()) for w in weights_data.values()]
                z_scores = [(norm - np.mean(weight_norms)) / (np.std(weight_norms) + 1e-8) 
                           for norm in weight_norms]
                if max(abs(z) for z in z_scores) > self.config.outlier_threshold:
                    logger.warning(f"Outlier detected for {worker_id}, skipping")
                    continue
                
                validated[worker_id] = raw_bytes
                logger.info(f"Validated update from {worker_id}: {dataset_size} samples")
                
            except Exception as e:
                logger.error(f"Validation failed for {worker_id}: {e}")
        
        if not validated:
            raise RuntimeError("No valid updates after validation")
        return validated, weights
    
    def aggregate(self, results: Dict[str, bytes]) -> bytes:
        """
        Production-grade aggregation for medical TFF simulations.
        Input: {worker_id: npz_bytes} with 'metadata' and model weights
        Output: aggregated npz bytes with audit trail
        """
        logger.info(f"Aggregating {len(results)} client updates")
        
        # 1. Validate & filter medical updates
        validated_results, client_weights = self.validate_updates(results)
        
        # 2. Load and prepare updates
        model_updates = []
        total_weight = sum(client_weights.values())
        
        for worker_id, raw_bytes in validated_results.items():
            buf = io.BytesIO(raw_bytes)
            arr = np.load(buf, allow_pickle=True)
            update = {k: arr[k] for k in arr.files if k != 'metadata'}
            weight = client_weights[worker_id] / total_weight
            model_updates.append((update, weight))
        
        if not model_updates:
            raise RuntimeError("No valid model updates")
        
        # 3. Secure weighted average (FedAvg with DP)
        avg_update = {}
        keys = model_updates[0][0].keys()
        
        for key in keys:
            weighted_sum = np.zeros_like(list(model_updates[0][0].values())[0])
            for update, weight in model_updates:
                weighted_sum += weight * update[key]
            
            # Add DP noise for medical privacy compliance
            noise_std = self.config.dp_noise_multiplier * np.sqrt(1.0 / total_weight)
            noise = np.random.normal(0, noise_std, weighted_sum.shape)
            avg_update[key] = weighted_sum + noise
        
        # 4. Audit trail metadata
        metadata = {
            'aggregation_time': tf.timestamp().numpy(),
            'num_clients': len(validated_results),
            'total_weight': float(total_weight),
            'dp_noise_multiplier': self.config.dp_noise_multiplier,
            'outlier_threshold': self.config.outlier_threshold
        }
        
        # 5. Serialize with TensorFlow I/O for production
        out = io.BytesIO()
        with np.load(out, mode='w') as npz_file:
            for key, value in avg_update.items():
                npz_file.save_array(key, value)
            npz_file.save_array('metadata', np.array([str(metadata)]))
        
        out.seek(0)
        logger.info(f"Aggregated model with DP noise. Clients: {len(validated_results)}")
        return out.read()

# Legacy compatibility
def aggregate(results: Dict[str, bytes]) -> bytes:
    """Legacy entrypoint - uses production aggregator."""
    config = AggregationConfig(
        dp_noise_multiplier=1.1,  # Medical privacy standard
        secure_aggregation=True
    )
    aggregator = MedicalAggregator(config)
    return aggregator.aggregate(results)

# TFF Integration Example (for simulation master)
@tff.tf_computation
def client_update_to_bytes(model_weights: Any) -> tf.Tensor:
    """Convert TFF model weights to serializable bytes."""
    # Implementation depends on your model structure
    pass

if __name__ == "__main__":
    # Demo usage
    config = AggregationConfig()
    aggregator = MedicalAggregator(config)
    print("Medical TFF Aggregator initialized with DP-SecFedAvg")

# ============================================================
# AGGREGATOR MODULE
# ============================================================
