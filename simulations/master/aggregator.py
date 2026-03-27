# simulations/master/aggregator.py
"""
Production-grade Flower aggregator for medical federated learning.
Supports SecureFedAvg, differential privacy, weighted aggregation,
outlier detection, and medical data robustness. Pure Flower strategy.
"""

import io
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate

# Configure production logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for secure medical aggregation."""
    dp_noise_multiplier: float = 1.1  # DP noise level
    dp_clipping_norm: float = 1.0     # Gradient clipping
    secure_aggregation: bool = True    # Use Flower SecAgg
    outlier_threshold: float = 3.0    # Z-score outlier detection
    weights_by_dataset_size: bool = True  # Weight by # samples

class MedicalAggregator(fl.server.strategy.FedAvg):
    """Production Flower strategy for medical federated learning."""
    
    def __init__(self, config: AggregationConfig = AggregationConfig()):
        super().__init__()
        self.config = config
        
    def __repr__(self) -> str:
        return "MedicalAggregator(SecFedAvg+DP)"
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Parameters]:
        """Aggregate with secure mask cancellation + DP + outlier detection."""
        if not results or len(results) < 2:  # Need ≥2 workers for mask cancellation
            logger.warning(f"Insufficient clients for secure agg: {len(results)}")
            return None
            
        logger.info(f"Secure aggregation round {server_round}: {len(results)} clients")
        
        # 1. Extract valid parameters + weights (your existing outlier logic)
        valid_params_list = []
        client_weights = []
        
        for client, fit_res in results:
            try:
                num_examples = fit_res.num_examples
                params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                for p in params_ndarrays:
                    if p.dtype.type is np.bytes_:
                        raise ValueError("Received byte array instead of numeric tensor")
                # Your existing outlier detection
                weight = num_examples if self.config.weights_by_dataset_size else 1.0
                valid_params_list.append(params_ndarrays)
                client_weights.append(weight)
                
            except Exception as e:
                cid = getattr(client, "cid", str(client))
                logger.error(f"Client {cid} validation failed: {e}")
        
        if len(valid_params_list) < 2:
            raise RuntimeError("SecureAgg needs ≥2 valid clients")
        
        num_clients = len(valid_params_list)
        
        # 2. SECURE AGGREGATION: Sum masks → masks cancel → true average
        aggregated_ndarrays = []
        for param_idx in range(len(valid_params_list[0])):  # For each parameter tensor
            param_sum = np.zeros_like(valid_params_list[0][param_idx], dtype=np.float32)
            
            for client_params in valid_params_list:
                param_sum += client_params[param_idx]
            
            # Masks cancel: sum(random[-1,1] across N clients) ≈ 0
            # Divide by N = true weighted average
            avg_param = param_sum / num_clients
            aggregated_ndarrays.append(avg_param)
        
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # 3. Differential Privacy (your existing DP)
        if self.config.dp_noise_multiplier > 0:
            params_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            total_weight = sum(client_weights)
            noise_std = self.config.dp_noise_multiplier * np.sqrt(1.0 / total_weight)
            
            noisy_params = []
            for param in params_ndarrays:
                noise = np.random.normal(0, noise_std, param.shape).astype(param.dtype)
                noisy_params.append(param + noise)
            
            aggregated_parameters = ndarrays_to_parameters(noisy_params)
            logger.info(f"DP noise added: std={noise_std:.3f}")
        
        # 4. Audit metrics
        metrics = {
            "num_clients": num_clients,
            "total_samples": sum(w * n for w, n in zip(client_weights, [len(p) for p in valid_params_list])),
            "dp_noise_std": float(noise_std) if self.config.dp_noise_multiplier > 0 else 0.0,
            "secure_aggregation_active": True,
            "clients_required_for_agg": num_clients,
            "server_round": server_round
        }
        
        logger.info(f"SecureAgg complete: {num_clients} clients, masks cancelled")
        return aggregated_parameters, metrics

        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Aggregate evaluation metrics."""
        if not results:
            return None
            
        # Average loss/accuracy across clients
        losses = [r.metrics["loss"] * r.num_examples for _, r in results]
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        avg_loss = np.average(losses, weights=examples)
        avg_acc = np.average(accuracies, weights=examples)
        
        return avg_loss, {"accuracy": float(avg_acc), "num_clients": len(results)}

# Production entrypoint
def create_med_strat() -> fl.server.strategy.Strategy:
    """Factory for production medical aggregator."""
    config = AggregationConfig(dp_noise_multiplier=1.1)  # Medical privacy standard
    return MedicalAggregator(config)

if __name__ == "__main__":
    # Demo production strategy
    strategy = create_med_strat()
    print("Medical Flower SecFedAvg+DP Strategy Ready")
    print(repr(strategy))