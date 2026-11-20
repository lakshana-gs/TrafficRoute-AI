"""P4 Spatio-Temporal Traffic Speed Prediction Package.

Implements Diffusion Convolutional Recurrent Neural Network (DCRNN) and
Temporal-only baseline for authentic METR-LA traffic forecasting.
"""

from src.models.p4.diffusion import (
    GraphProperties,
    build_diffusion_supports_from_pkl,
    compute_diffusion_supports,
    normalize_directed_adjacency,
    validate_sensor_graph,
)
from src.models.p4.dcrnn import (
    DCGRUCell,
    DCRNNDecoder,
    DCRNNEncoder,
    DCRNNModel,
    DiffusionGraphConv,
)
from src.models.p4.temporal_only import TemporalOnlyGRU
from src.models.p4.dataset import (
    MaskedMAELoss,
    PyTorchWindowDataset,
    create_traffic_dataloaders,
    masked_mae_loss_tensor,
)
from src.models.p4.trainer import ModelTrainer, TrainingConfig, TrainingHistory
from src.models.p4.evaluator import (
    P4Evaluator,
    P4EvaluationResults,
    compute_missing_data_statistics,
    compute_sensor_level_metrics,
)

__all__ = [
    "GraphProperties",
    "validate_sensor_graph",
    "normalize_directed_adjacency",
    "compute_diffusion_supports",
    "build_diffusion_supports_from_pkl",
    "DiffusionGraphConv",
    "DCGRUCell",
    "DCRNNEncoder",
    "DCRNNDecoder",
    "DCRNNModel",
    "TemporalOnlyGRU",
    "MaskedMAELoss",
    "PyTorchWindowDataset",
    "create_traffic_dataloaders",
    "masked_mae_loss_tensor",
    "ModelTrainer",
    "TrainingConfig",
    "TrainingHistory",
    "P4Evaluator",
    "P4EvaluationResults",
    "compute_sensor_level_metrics",
    "compute_missing_data_statistics",
]
