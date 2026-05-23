"""
Baseline models stuff:
   TransformerIDS  >>>standard encoder + mean-pool + MLP head
   LSTMIDS  >>> bidirectional LSTM + last-hidden + MLP head

Both takes (B, n_features) input as MambaIDS / RWKVIDS

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerIDS(nn.Module):


    def __init__(
        self,
        n_features:  int,
        num_classes: int,
        d_model:     int   = 256,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        ff_dim:      int   = 512,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_features  = n_features
        self.d_model     = d_model

        self.input_proj = nn.Linear(1, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

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
        x = x + self.pos_emb
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LSTMIDS(nn.Module):


    def __init__(
        self,
        n_features:   int,
        num_classes:  int,
        hidden_size:  int   = 128,
        n_layers:     int   = 2,
        dropout:      float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        dirs = 2 if bidirectional else 1

        self.input_proj = nn.Linear(1, hidden_size)

        self.lstm = nn.LSTM(
            input_size   = hidden_size,
            hidden_size  = hidden_size,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * dirs),
            nn.Linear(hidden_size * dirs, hidden_size * dirs * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * dirs * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        out, (h, _) = self.lstm(x)

        if self.bidirectional:

            fwd = h[-2]
            bwd = h[-1]
            feat = torch.cat([fwd, bwd], dim=-1)
        else:
            feat = h[-1]

        return self.classifier(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
