"""
RWKV-IDS  —  Linear Attention RNN for Network Intrusion Detection.

"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def wkv_parallel(
    w: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:

    B, L, D = k.shape
    device   = k.device
    dtype    = k.dtype

    w_neg = -w.abs()

    ys = []
    num = torch.zeros(B, D, device=device, dtype=dtype)
    den = torch.zeros(B, D, device=device, dtype=dtype)

    for t in range(L):

        k_t = k[:, t, :]
        v_t = v[:, t, :]

        eu  = torch.exp(u + k_t)
        ew  = torch.exp(w_neg)

        num = ew * num + eu * v_t
        den = ew * den + eu

        wkv_t = num / (den + 1e-30)
        ys.append(wkv_t)

    return torch.stack(ys, dim=1)

def wkv_step(
    w:   torch.Tensor,
    u:   torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    num: torch.Tensor,
    den: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    ew  = torch.exp(-w.abs())
    eu  = torch.exp(u + k_t)

    new_num = ew * num + eu * v_t
    new_den = ew * den + eu
    wkv_t   = new_num / (new_den + 1e-30)
    return wkv_t, new_num, new_den

class TimeMixing(nn.Module):


    def __init__(self, d_model: int, layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model

        ratio = layer_idx / max(1, 4)
        self.time_mix_k = nn.Parameter(
            torch.ones(1, 1, d_model) * (1 - ratio * 0.1)
        )
        self.time_mix_v = nn.Parameter(
            torch.ones(1, 1, d_model) * (1 - ratio * 0.2)
        )
        self.time_mix_r = nn.Parameter(
            torch.ones(1, 1, d_model) * (1 - ratio * 0.5)
        )

        w_init = torch.zeros(d_model)
        for i in range(d_model):
            w_init[i] = -5 + 8 * (i / (d_model - 1)) ** (
                0.7 + 1.3 * i / (d_model - 1)
            )
        self.time_decay  = nn.Parameter(w_init)
        self.time_first  = nn.Parameter(
            torch.zeros(d_model).uniform_(-1, 1)
        )

        self.key        = nn.Linear(d_model, d_model, bias=False)
        self.value      = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output     = nn.Linear(d_model, d_model, bias=False)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        residual = x
        x_n = self.norm(x)

        x_shifted = F.pad(x_n, (0, 0, 1, 0))[:, :-1, :]

        xk = x_n * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        xv = x_n * self.time_mix_v + x_shifted * (1 - self.time_mix_v)
        xr = x_n * self.time_mix_r + x_shifted * (1 - self.time_mix_r)

        k   = self.key(xk)
        v   = self.value(xv)
        r   = self.receptance(xr)

        wkv  = wkv_parallel(self.time_decay, self.time_first, k, v)
        rwkv = torch.sigmoid(r) * wkv
        out  = self.output(rwkv)
        return out + residual

    def forward_step(
        self,
        x_t:    torch.Tensor,
        x_prev: torch.Tensor,
        num:    torch.Tensor,
        den:    torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      
        x_n = self.norm(x_t)

        xk = x_n * self.time_mix_k.squeeze() + x_prev * (1 - self.time_mix_k.squeeze())
        xv = x_n * self.time_mix_v.squeeze() + x_prev * (1 - self.time_mix_v.squeeze())
        xr = x_n * self.time_mix_r.squeeze() + x_prev * (1 - self.time_mix_r.squeeze())

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        wkv_t, new_num, new_den = wkv_step(
            self.time_decay, self.time_first, k, v, num, den
        )
        rwkv = torch.sigmoid(r) * wkv_t
        out  = self.output(rwkv) + x_t
        return out, x_n, new_num, new_den

    def init_state(self, batch_size: int, device: torch.device):
        
        return (
            torch.zeros(batch_size, self.d_model, device=device),
            torch.zeros(batch_size, self.d_model, device=device),
            torch.zeros(batch_size, self.d_model, device=device),
        )

class ChannelMixing(nn.Module):
   

    def __init__(self, d_model: int, ff_dim: Optional[int] = None):
        super().__init__()
        ff_dim = ff_dim or d_model * 4

        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        self.key        = nn.Linear(d_model, ff_dim,  bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value      = nn.Linear(ff_dim,  d_model, bias=False)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        residual = x
        x_n = self.norm(x)

        x_shifted = F.pad(x_n, (0, 0, 1, 0))[:, :-1, :]

        xk = x_n * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        xr = x_n * self.time_mix_r + x_shifted * (1 - self.time_mix_r)

        k   = self.key(xk)
        r   = self.receptance(xr)
        kv  = torch.relu(k) ** 2
        out = torch.sigmoid(r) * self.value(kv)
        return out + residual

    def forward_step(
        self,
        x_t:    torch.Tensor,
        x_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x_n  = self.norm(x_t)
        xk   = x_n * self.time_mix_k.squeeze() + x_prev * (1 - self.time_mix_k.squeeze())
        xr   = x_n * self.time_mix_r.squeeze() + x_prev * (1 - self.time_mix_r.squeeze())
        k    = self.key(xk)
        r    = self.receptance(xr)
        kv   = torch.relu(k) ** 2
        out  = torch.sigmoid(r) * self.value(kv) + x_t
        return out, x_n

class RWKVBlock(nn.Module):
    def __init__(self, d_model: int, layer_idx: int = 0, dropout: float = 0.0):
        super().__init__()
        self.time_mix    = TimeMixing(d_model, layer_idx)
        self.channel_mix = ChannelMixing(d_model)
        self.drop        = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mix(x)
        x = self.channel_mix(x)
        return self.drop(x)

    def forward_step(self, x_t, tm_state, cm_x_prev):
        
        tm_x_prev, num, den = tm_state
        x_t, new_tm_x_prev, new_num, new_den = self.time_mix.forward_step(
            x_t, tm_x_prev, num, den
        )
        x_t, new_cm_x_prev = self.channel_mix.forward_step(x_t, cm_x_prev)
        return x_t, (new_tm_x_prev, new_num, new_den), new_cm_x_prev

    def init_state(self, batch_size, device):
        return (
            self.time_mix.init_state(batch_size, device),
            torch.zeros(batch_size, self.time_mix.d_model, device=device),
        )

class RWKVIDS(nn.Module):


    def __init__(
        self,
        n_features:  int,
        num_classes: int,
        d_model:     int   = 256,
        n_layers:    int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_features  = n_features
        self.d_model     = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Linear(1, d_model)

        self.blocks = nn.ModuleList([
            RWKVBlock(d_model, layer_idx=i, dropout=dropout)
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

    def init_rnn_state(self, batch_size: int, device: torch.device):
        
        return [b.init_state(batch_size, device) for b in self.blocks]

    def forward_rnn(
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
            for block, (tm_state, cm_x_prev) in zip(self.blocks, state):
                token, new_tm_state, new_cm_x_prev = block.forward_step(
                    token, tm_state, cm_x_prev
                )
                new_block_states.append((new_tm_state, new_cm_x_prev))
            state = new_block_states

        return self.classifier(token), state

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
