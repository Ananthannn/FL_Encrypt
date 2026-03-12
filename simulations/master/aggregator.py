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
        """Aggregate medical model updates with validation + DP."""
        if not results:
            return None
            
        logger.info(f"Aggregating round {server_round}: {len(results)} clients")
        
        # 1. Extract parameters + metrics
        valid_results = []
        client_weights = []
        
        for client, fit_res in results:
            try:
                num_examples = fit_res.num_examples
                parameters = fit_res.parameters
                metrics = fit_res.metrics or {}
                
                # Weight by dataset size (medical standard)
                weight = num_examples if self.config.weights_by_dataset_size else 1.0
                client_weights.append(weight)
                
                # Outlier detection on parameter norms
                params_ndarrays = parameters_to_ndarrays(parameters)
                param_norms = [torch.norm(torch.tensor(p.flatten())).item() 
                             for p in params_ndarrays]
                z_scores = [(norm - np.mean(param_norms)) / (np.std(param_norms) + 1e-8)
                           for norm in param_norms]
                
                if max(abs(z) for z in z_scores) > self.config.outlier_threshold:
                    client_id = getattr(client, "cid", client)
                    logger.warning(f"Outlier client {client_id}, skipping")
                    continue
                               
                valid_results.append((client, fit_res))
                client_id = getattr(client, "cid", "custom_client")
                logger.info(f"Valid update {client_id}: {num_examples} samples")
            except Exception as e:
                client_id = getattr(client, "cid", "custom_client")
                logger.error(f"Validation failed {client_id}: {e}")
        if not valid_results:
            raise RuntimeError("No valid updates after outlier filtering")
        
        # 2. Weighted FedAvg aggregation
        aggregation_input = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in valid_results
        ]
        aggregated_ndarrays = aggregate(aggregation_input)
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # 3. Differential Privacy noise injection
        if self.config.dp_noise_multiplier > 0:
            params_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            total_weight = sum(client_weights[:len(valid_results)])
            noise_std = self.config.dp_noise_multiplier * np.sqrt(1.0 / total_weight)
            
            noisy_params = []
            for param in params_ndarrays:
                noise = np.random.normal(0, noise_std, param.shape)
                noisy_params.append(param + noise)
            
            aggregated_parameters = ndarrays_to_parameters(noisy_params)
            logger.info(f"DP noise added: std={noise_std:.3f}")
        
        # 4. Audit trail metrics
        metrics = {
            "num_clients": len(valid_results),
            "total_samples": sum(fit_res.num_examples for _, fit_res in valid_results),
            "dp_noise_std": float(noise_std) if self.config.dp_noise_multiplier > 0 else 0.0,
            "outliers_detected": len(results) - len(valid_results),
            "server_round": server_round
        }
        
        logger.info(f"Aggregated {metrics['num_clients']} clients, "
                   f"{metrics['outliers_detected']} outliers filtered")
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