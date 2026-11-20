"""Diffusion Convolutional Recurrent Neural Network (DCRNN) Architecture.

Reference:
    Li et al., "Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting", ICLR 2018.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class DiffusionGraphConv(nn.Module):
    """
    Diffusion Graph Convolution Layer.
    
    Operates on graph signals X of shape (B, N, C_in) with S diffusion support matrices:
        Z = sum_{k=0}^{S-1} (P_k X W_k) + b
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_supports: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_supports = num_supports
        
        self.weight = nn.Parameter(torch.empty(num_supports * in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        supports: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with optimized single-matmul support aggregation.
        
        Args:
            x: Input tensor of shape (B, N, C_in)
            supports: List of S support matrices of shape (N, N), or concatenated tensor (S*N, N)
            
        Returns:
            Output tensor of shape (B, N, C_out)
        """
        B, N, C = x.shape
        K = self.num_supports

        if isinstance(supports, list):
            supports_cat = torch.cat(supports, dim=0)
        else:
            supports_cat = supports

        x_flat = x.permute(1, 0, 2).reshape(N, B * C)
        sx = torch.matmul(supports_cat, x_flat)  # (K*N, B*C)
        sx = sx.view(K, N, B, C).permute(2, 1, 0, 3).reshape(B, N, K * C)
        out = torch.matmul(sx, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional Gated Recurrent Unit (DCGRU) Cell.
    
    Replaces matrix multiplication in standard GRU with diffusion graph convolutions.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_supports: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_supports = num_supports

        # Gate convolutions (reset + update combined for computational efficiency)
        self.conv_gates = DiffusionGraphConv(
            in_features=in_features + hidden_dim,
            out_features=2 * hidden_dim,
            num_supports=num_supports,
        )
        
        # Candidate state convolution
        self.conv_candidate = DiffusionGraphConv(
            in_features=in_features + hidden_dim,
            out_features=hidden_dim,
            num_supports=num_supports,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        supports: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute one DCGRU step.
        
        Args:
            x: Current input at step t, shape (B, N, C_in)
            h_prev: Previous hidden state, shape (B, N, hidden_dim)
            supports: Graph diffusion supports
            
        Returns:
            New hidden state of shape (B, N, hidden_dim)
        """
        combined = torch.cat([x, h_prev], dim=-1)
        gates = torch.sigmoid(self.conv_gates(combined, supports))
        r, u = torch.chunk(gates, 2, dim=-1)

        cand_input = torch.cat([x, r * h_prev], dim=-1)
        cand = torch.tanh(self.conv_candidate(cand_input, supports))

        h_new = u * h_prev + (1.0 - u) * cand
        return h_new


class DCRNNEncoder(nn.Module):
    """
    Multi-layer DCGRU Encoder.
    
    Processes the historical sequence of length T_in and produces final hidden states.
    """
    def __init__(
        self,
        in_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_supports: int = 5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList()
        
        for i in range(num_layers):
            c_in = in_features if i == 0 else hidden_dim
            self.cells.append(DCGRUCell(c_in, hidden_dim, num_supports))

    def forward(
        self,
        x: torch.Tensor,
        supports: Union[List[torch.Tensor], torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Args:
            x: Historical input tensor of shape (B, T_in, N, C_in)
            supports: Graph diffusion supports
            
        Returns:
            List of final hidden state tensors, each of shape (B, N, hidden_dim)
        """
        B, T_in, N, _ = x.shape
        hidden_states = [
            torch.zeros(B, N, self.hidden_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]

        # Pre-concatenate supports once if passed as list
        supports_cat = torch.cat(supports, dim=0) if isinstance(supports, list) else supports

        for t in range(T_in):
            cur_in = x[:, t]
            new_hidden_states = []
            for layer_idx, cell in enumerate(self.cells):
                h_new = cell(cur_in, hidden_states[layer_idx], supports_cat)
                new_hidden_states.append(h_new)
                cur_in = h_new
            hidden_states = new_hidden_states

        return hidden_states


class DCRNNDecoder(nn.Module):
    """
    Multi-layer DCGRU Decoder.
    
    Generates multi-step forecasts initialized with encoder hidden states.
    """
    def __init__(
        self,
        out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_supports: int = 5,
    ):
        super().__init__()
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        
        for i in range(num_layers):
            c_in = out_features if i == 0 else hidden_dim
            self.cells.append(DCGRUCell(c_in, hidden_dim, num_supports))
            
        self.out_proj = nn.Linear(hidden_dim, out_features)

    def forward(
        self,
        init_input: torch.Tensor,
        hidden_states: List[torch.Tensor],
        horizon: int,
        supports: Union[List[torch.Tensor], torch.Tensor],
        y_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            init_input: Initial decoder input (e.g. x[:, -1]), shape (B, N, C_out)
            hidden_states: Initialized hidden states from encoder
            horizon: Prediction horizon steps (default: 12)
            supports: Graph diffusion supports
            y_target: Optional ground-truth targets for teacher forcing, shape (B, horizon, N, C_out)
            teacher_forcing_ratio: Probability of using ground truth input during training
            
        Returns:
            Forecast tensor of shape (B, horizon, N, C_out)
        """
        outputs = []
        cur_input = init_input
        supports_cat = torch.cat(supports, dim=0) if isinstance(supports, list) else supports

        for t in range(horizon):
            new_hidden_states = []
            cur_layer_in = cur_input
            for layer_idx, cell in enumerate(self.cells):
                h_new = cell(cur_layer_in, hidden_states[layer_idx], supports_cat)
                new_hidden_states.append(h_new)
                cur_layer_in = h_new
            hidden_states = new_hidden_states

            out_step = self.out_proj(cur_layer_in)  # (B, N, C_out)
            outputs.append(out_step)

            if (
                y_target is not None
                and teacher_forcing_ratio > 0.0
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                cur_input = y_target[:, t]
            else:
                cur_input = out_step

        return torch.stack(outputs, dim=1)


class DCRNNModel(nn.Module):
    """
    End-to-End Sequence-to-Sequence DCRNN Model.
    
    Maps history (B, 12, N, 1) to horizon (B, 12, N, 1) using diffusion graph convolutions.
    """
    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_supports: int = 5,
        horizon: int = 12,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_supports = num_supports
        self.horizon = horizon

        self.encoder = DCRNNEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_supports=num_supports,
        )
        self.decoder = DCRNNDecoder(
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_supports=num_supports,
        )

    def forward(
        self,
        x: torch.Tensor,
        supports: Union[List[torch.Tensor], torch.Tensor],
        y_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Execute full sequence-to-sequence prediction.
        
        Args:
            x: Input speed tensor of shape (B, 12, N, 1)
            supports: List of diffusion support tensors
            y_target: Optional target tensor for training teacher forcing
            teacher_forcing_ratio: Scheduled sampling probability
            
        Returns:
            Predicted speed tensor of shape (B, 12, N, 1)
        """
        supports_cat = torch.cat(supports, dim=0) if isinstance(supports, list) else supports
        enc_hidden = self.encoder(x, supports_cat)
        init_dec_input = x[:, -1]
        out = self.decoder(
            init_input=init_dec_input,
            hidden_states=enc_hidden,
            horizon=self.horizon,
            supports=supports_cat,
            y_target=y_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return out
