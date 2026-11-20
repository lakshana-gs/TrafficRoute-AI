"""Adjacency matrix validation, directed normalization, and diffusion support construction."""

import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import torch


class GraphProperties(NamedTuple):
    """Detailed structural metrics of the METR-LA sensor graph."""
    sensor_count: int
    adjacency_shape: Tuple[int, int]
    is_finite: bool
    has_nan: bool
    has_inf: bool
    min_weight: float
    max_weight: float
    diagonal_min: float
    diagonal_max: float
    is_symmetric: bool
    nonzero_edges: int
    total_entries: int
    density: float


def validate_sensor_graph(
    adj_mx: np.ndarray,
    sensor_ids: Optional[List[str]] = None,
) -> GraphProperties:
    """
    Perform comprehensive validation of the sensor adjacency matrix.
    
    Verifies node count, finite status, diagonal self-loops, asymmetry, and density.
    """
    adj = np.asarray(adj_mx, dtype=np.float64)
    N, M = adj.shape
    if N != M:
        raise ValueError(f"Adjacency matrix must be square, got shape ({N}, {M}).")
    
    if sensor_ids is not None and len(sensor_ids) != N:
        raise ValueError(f"Sensor ID count ({len(sensor_ids)}) does not match adjacency size ({N}).")

    is_finite = bool(np.all(np.isfinite(adj)))
    has_nan = bool(np.any(np.isnan(adj)))
    has_inf = bool(np.any(np.isinf(adj)))
    min_weight = float(np.min(adj))
    max_weight = float(np.max(adj))
    diag = np.diag(adj)
    diag_min = float(np.min(diag))
    diag_max = float(np.max(diag))
    is_symmetric = bool(np.allclose(adj, adj.T, atol=1e-6))
    nonzero_edges = int(np.count_nonzero(adj))
    total_entries = int(adj.size)
    density = float(nonzero_edges / total_entries)

    return GraphProperties(
        sensor_count=N,
        adjacency_shape=(N, M),
        is_finite=is_finite,
        has_nan=has_nan,
        has_inf=has_inf,
        min_weight=round(min_weight, 6),
        max_weight=round(max_weight, 6),
        diagonal_min=round(diag_min, 6),
        diagonal_max=round(diag_max, 6),
        is_symmetric=is_symmetric,
        nonzero_edges=nonzero_edges,
        total_entries=total_entries,
        density=round(density, 6),
    )


def normalize_directed_adjacency(adj_mx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward (downstream) and backward (upstream) random-walk transition matrices.
    
    Formulas:
        Forward transition:  P_f = D_out^{-1} A,   where D_out,ii = sum_j A_ij
        Backward transition: P_b = D_in^{-1} A^T, where D_in,jj  = sum_i A_ij
        
    Args:
        adj_mx: (N, N) raw adjacency matrix
        
    Returns:
        Tuple of (P_forward, P_backward), each of shape (N, N).
    """
    adj = np.asarray(adj_mx, dtype=np.float32)
    N = adj.shape[0]

    # Forward random-walk (row normalization)
    d_out = np.sum(adj, axis=1)
    d_out_inv = np.zeros(N, dtype=np.float32)
    pos_mask_out = d_out > 0.0
    d_out_inv[pos_mask_out] = 1.0 / d_out[pos_mask_out]
    P_f = np.diag(d_out_inv) @ adj

    # Backward random-walk (column normalization / transpose row normalization)
    d_in = np.sum(adj, axis=0)
    d_in_inv = np.zeros(N, dtype=np.float32)
    pos_mask_in = d_in > 0.0
    d_in_inv[pos_mask_in] = 1.0 / d_in[pos_mask_in]
    P_b = np.diag(d_in_inv) @ adj.T

    return P_f, P_b


def compute_diffusion_supports(
    adj_mx: np.ndarray,
    k: int = 2,
) -> List[np.ndarray]:
    """
    Compute diffusion polynomial transition support matrices up to order K.
    
    Supports list contains:
        [I, P_f, P_f^2, ..., P_f^K, P_b, P_b^2, ..., P_b^K]
        
    Total number of supports = 2 * K + 1.
    For K=2, yields 5 matrices of shape (N, N).
    """
    if k < 1:
        raise ValueError(f"Diffusion step K must be >= 1, got {k}.")
        
    N = adj_mx.shape[0]
    P_f, P_b = normalize_directed_adjacency(adj_mx)
    
    supports: List[np.ndarray] = [np.eye(N, dtype=np.float32)]
    
    # Forward powers
    cur_pf = P_f.copy()
    supports.append(cur_pf)
    for _ in range(2, k + 1):
        cur_pf = cur_pf @ P_f
        supports.append(cur_pf)
        
    # Backward powers
    cur_pb = P_b.copy()
    supports.append(cur_pb)
    for _ in range(2, k + 1):
        cur_pb = cur_pb @ P_b
        supports.append(cur_pb)
        
    return supports


def save_normalized_supports(
    supports: List[np.ndarray],
    output_npz_path: str | Path = "data/traffic/processed/metr_la/normalized_adj_mx.npz",
    metadata_json_path: Optional[str | Path] = "data/traffic/processed/metr_la/normalized_adj_meta.json",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save normalized support matrices and documentation metadata to processed dir."""
    out_npz = Path(output_npz_path)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    
    supports_dict = {f"support_{i}": s for i, s in enumerate(supports)}
    np.savez_compressed(out_npz, **supports_dict)

    if metadata_json_path:
        out_json = Path(metadata_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "dataset": "METR-LA",
            "num_sensors": supports[0].shape[0],
            "num_supports": len(supports),
            "diffusion_order_k": (len(supports) - 1) // 2,
            "normalization_formula": "P_f = D_out^{-1} A, P_b = D_in^{-1} A^T",
            "support_names": [
                "Identity (order 0)",
                *[f"P_forward^({p})" for p in range(1, (len(supports) - 1) // 2 + 1)],
                *[f"P_backward^({p})" for p in range(1, (len(supports) - 1) // 2 + 1)],
            ],
            "extra_metadata": extra_meta or {},
        }
        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)


def build_diffusion_supports_from_pkl(
    pkl_path: str | Path = "data/traffic/raw/metr_la/adj_mx.pkl",
    k: int = 2,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Load raw sensor graph and construct PyTorch tensor diffusion supports.
    """
    path = Path(pkl_path)
    if not path.exists():
        raise FileNotFoundError(f"Adjacency matrix file not found: {path}")

    with open(path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    supports_np = compute_diffusion_supports(adj_mx, k=k)
    supports_tensor = [torch.from_numpy(s).float().to(device) for s in supports_np]
    return supports_tensor
