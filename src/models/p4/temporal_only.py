"""Temporal-Only GRU Baseline for Spatial Ablation Experiment.

Implements a purely temporal recurrent architecture with identical capacity (layers, hidden dim)
to DCRNN, but completely excluding the spatial graph adjacency matrix.
"""

from typing import Optional
import torch
import torch.nn as nn


class TemporalOnlyGRU(nn.Module):
    """
    Temporal-only GRU sequence-to-sequence model.
    
    Processes the time series of each sensor independently across history and horizon,
    providing a clean spatial ablation baseline.
    """
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 12,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.encoder_gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder_gru = nn.GRU(
            input_size=out_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, out_features)

    def forward(
        self,
        x: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: Input speed tensor of shape (B, 12, N, 1)
            y_target: Optional target tensor for training teacher forcing
            teacher_forcing_ratio: Scheduled sampling probability
            
        Returns:
            Predicted speed tensor of shape (B, 12, N, 1)
        """
        B, T_in, N, C = x.shape
        # Flatten batch and sensor dimensions so each sensor is an independent temporal sequence
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T_in, C)  # (B*N, 12, 1)

        _, h_enc = self.encoder_gru(x_flat)  # h_enc: (num_layers, B*N, hidden_dim)

        cur_input = x_flat[:, -1:, :]  # Initial decoder input: (B*N, 1, 1)
        h_dec = h_enc
        outputs = []

        if y_target is not None and teacher_forcing_ratio > 0.0:
            y_target_flat = y_target.permute(0, 2, 1, 3).reshape(B * N, self.horizon, self.out_features)
        else:
            y_target_flat = None

        for t in range(self.horizon):
            out_step, h_dec = self.decoder_gru(cur_input, h_dec)  # (B*N, 1, hidden_dim)
            pred_step = self.out_proj(out_step)  # (B*N, 1, out_features)
            outputs.append(pred_step)

            if (
                y_target_flat is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                cur_input = y_target_flat[:, t : t + 1, :]
            else:
                cur_input = pred_step

        out_seq = torch.cat(outputs, dim=1)  # (B*N, horizon, out_features)
        # Reshape back to (B, horizon, N, out_features)
        out = out_seq.reshape(B, N, self.horizon, self.out_features).permute(0, 2, 1, 3)
        return out
