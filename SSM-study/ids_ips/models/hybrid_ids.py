"""
Hybrid-IDS  —  Mamba blocks + RWKV channel-mixing .

"trying to investigate a hybrid architecture combining Mamba blocks 
(for long-range temporal patterns) with RWKV
channel-mixing, potentially capturing
complementary strengths of both approaches."

Design
------
Each hybrid block stacks:
  -> MambaBlock   — selective SSM for long-range sequence dependencies
  -> ChannelMixing — RWKV squared-ReLU gate for per-step feature interactions


"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ids_ips.models.mamba_ids import MambaBlock
from ids_ips.models.rwkv_ids  import ChannelMixing

class HybridBlock(nn.Module):


    def __init__(
        self,
        d_model: int,
        d_state: int  = 16,
        d_conv:  int  = 4,
        expand:  int  = 2,
        layer_idx: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mamba_block   = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.channel_mix   = ChannelMixing(d_model)
        self.drop          = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, d_model)"""
        x = self.mamba_block(x)
        x = self.channel_mix(x)
        return self.drop(x)

    def forward_step(
        self,
        x_t:       torch.Tensor,
        h_ssm:     torch.Tensor,
        conv_buf:  torch.Tensor,
        cm_x_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x_t, h_ssm, conv_buf = self.mamba_block.forward_step(x_t, h_ssm, conv_buf)
        x_t, cm_x_prev       = self.channel_mix.forward_step(x_t, cm_x_prev)
        return x_t, h_ssm, conv_buf, cm_x_prev

    def init_state(self, batch_size: int, device: torch.device):
        h_ssm, conv_buf = self.mamba_block.init_state(batch_size, device)
        cm_x_prev       = torch.zeros(
            batch_size, self.mamba_block.d_model, device=device
        )
        return h_ssm, conv_buf, cm_x_prev

class HybridIDS(nn.Module):


    def __init__(
        self,
        n_features:  int,
        num_classes: int,
        d_model:     int   = 256,
        n_layers:    int   = 4,
        d_state:     int   = 16,
        d_conv:      int   = 4,
        expand:      int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_features  = n_features
        self.d_model     = d_model
        self.num_classes = num_classes
        self.n_layers    = n_layers

        self.input_proj = nn.Linear(1, d_model)

        self.blocks = nn.ModuleList([
            HybridBlock(d_model, d_state, d_conv, expand,
                        layer_idx=i, dropout=dropout)
            for i in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def init_stream_state(self, batch_size: int, device: torch.device):
        
        return [b.init_state(batch_size, device) for b in self.blocks]

    def forward_stream(
        self,
        x:     torch.Tensor,
        state: List,
    ) -> Tuple[torch.Tensor, List]:
        
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        token = x[:, 0, :]
        for feat_idx in range(self.n_features):
            token = x[:, feat_idx, :]
            new_block_states = []
            for block, (h_ssm, conv_buf, cm_x_prev) in zip(self.blocks, state):
                token, h_ssm, conv_buf, cm_x_prev = block.forward_step(
                    token, h_ssm, conv_buf, cm_x_prev
                )
                new_block_states.append((h_ssm, conv_buf, cm_x_prev))
            state = new_block_states

        return self.classifier(token), state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
