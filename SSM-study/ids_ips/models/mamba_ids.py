"""
Mamba-IDS  —  Selective State Space Model for Network Intrusion Detection.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def selective_scan_seq(
    u:     torch.Tensor,
    delta: torch.Tensor,
    A:     torch.Tensor,
    B:     torch.Tensor,
    C:     torch.Tensor,
    D:     torch.Tensor,
) -> torch.Tensor:
    B_sz, L, d_inner = u.shape
    N = A.shape[1]

    dA = torch.exp(
        rearrange(delta, 'b l d -> b l d 1') *
        rearrange(A,     'd n -> 1 1 d n')
    )
    dB_u = (
        rearrange(delta, 'b l d -> b l d 1') *
        rearrange(B,     'b l n -> b l 1 n') *
        rearrange(u,     'b l d -> b l d 1')
    )

    log_dA      = torch.log(dA.clamp(min=1e-38))
    log_cdA_inc = torch.cumsum(log_dA, dim=1)
    log_P = F.pad(log_cdA_inc, (0, 0, 0, 0, 1, 0))
    log_P_s1 = log_P[:, 1:]
    log_P_t1 = log_P_s1

    f    = dB_u * torch.exp((-log_P_s1).clamp(max=80))
    cumF = torch.cumsum(f, dim=1)

    h = torch.exp(log_P_t1) * cumF

    y = torch.einsum('btn,btdn->btd', C, h)
    return y + u * rearrange(D, 'd -> 1 1 d')

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int  = 16,
        d_conv:  int  = 4,
        expand:  int  = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv  = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,
            groups       = self.d_inner,
            bias         = True,
        )

        self.x_proj  = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = -torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(
            self.d_inner, -1
        ).log()
        self.A_log = nn.Parameter(A)
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)

    def _ssm_params(self, x_conv: torch.Tensor):
        
        proj    = self.x_proj(x_conv)
        B_ssm   = proj[..., :self.d_state]
        C_ssm   = proj[..., self.d_state:2 * self.d_state]
        dt_raw  = proj[..., 2 * self.d_state:]
        dt      = F.softplus(self.dt_proj(dt_raw))
        A       = -torch.exp(self.A_log.float())
        return dt, A, B_ssm, C_ssm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        xc = rearrange(x_branch, 'b l d -> b d l')
        xc = self.conv1d(xc)[..., :xc.shape[-1]]
        xc = rearrange(xc, 'b d l -> b l d')
        xc = F.silu(xc)

        dt, A, B_ssm, C_ssm = self._ssm_params(xc)

        y = selective_scan_seq(xc, dt, A, B_ssm, C_ssm, self.D)

        y   = y * F.silu(z)
        out = self.out_proj(y)
        return self.drop(out) + residual

    def forward_step(
        self,
        x:         torch.Tensor,
        h_ssm:     torch.Tensor,
        conv_buf:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv_in = torch.cat([conv_buf, x_branch.unsqueeze(-1)], dim=-1)
        new_conv_buf = x_conv_in[:, :, 1:]

        weight = self.conv1d.weight.squeeze(1)
        xc     = (x_conv_in * weight.unsqueeze(0)).sum(-1)
        if self.conv1d.bias is not None:
            xc = xc + self.conv1d.bias
        xc = F.silu(xc)

        proj   = self.x_proj(xc)
        B_ssm  = proj[..., :self.d_state]
        C_ssm  = proj[..., self.d_state:2 * self.d_state]
        dt_raw = proj[..., 2 * self.d_state:]
        dt     = F.softplus(self.dt_proj(dt_raw))
        A      = -torch.exp(self.A_log.float())

        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0)
        )
        dB_u = (
            dt.unsqueeze(-1) *
            B_ssm.unsqueeze(1) *
            xc.unsqueeze(-1)
        )

        h_ssm_new = dA * h_ssm + dB_u
        y_ssm     = (h_ssm_new * C_ssm.unsqueeze(1)).sum(-1)
        y_ssm     = y_ssm + xc * self.D

        y   = y_ssm * F.silu(z)
        out = self.out_proj(y) + residual
        return out, h_ssm_new, new_conv_buf

    def init_state(self, batch_size: int, device: torch.device):
       
        h_ssm    = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        conv_buf = torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device)
        return h_ssm, conv_buf

class MambaIDS(nn.Module):


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
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
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
        x:      torch.Tensor,
        state:  List,
    ) -> Tuple[torch.Tensor, List]:
      
        B = x.shape[0]
        device = x.device

        feat_seq = x.unsqueeze(-1)
        feat_seq = self.input_proj(feat_seq)

        new_state = []
        h_out     = None

        for feat_idx in range(self.n_features):
            token = feat_seq[:, feat_idx, :]
            new_block_states = []
            for layer_idx, (block, (h_ssm, conv_buf)) in enumerate(
                zip(self.blocks, state)
            ):
                token, h_ssm_new, conv_buf_new = block.forward_step(
                    token, h_ssm, conv_buf
                )
                new_block_states.append((h_ssm_new, conv_buf_new))
            state   = new_block_states
            h_out   = token

        return self.classifier(h_out), state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
