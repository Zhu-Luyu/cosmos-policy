# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn


class ActionLatentCodec(nn.Module):
    """Maps low-dimensional action chunks to and from DiT-compatible latent frames."""

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
        bottleneck_dim: int = 512,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim

        self.action_numel = chunk_size * action_dim
        self.latent_numel = latent_channels * latent_height * latent_width

        self.encoder = nn.Sequential(
            nn.Linear(self.action_numel, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, self.latent_numel),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_numel, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_numel),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _check_action_shape(self, action_chunk: torch.Tensor) -> None:
        expected_shape = (self.chunk_size, self.action_dim)
        actual_shape = tuple(action_chunk.shape[-2:])
        if actual_shape != expected_shape:
            raise ValueError(f"Expected action chunk trailing shape {expected_shape}, got {actual_shape}.")

    def _check_frame_shape(self, action_latent_frame: torch.Tensor) -> None:
        expected_shape = (self.latent_channels, self.latent_height, self.latent_width)
        actual_shape = tuple(action_latent_frame.shape[-3:])
        if actual_shape != expected_shape:
            raise ValueError(f"Expected action latent frame trailing shape {expected_shape}, got {actual_shape}.")

    def encode_frame(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Encode an action chunk `(B, chunk_size, action_dim)` into `(B, C, H, W)`."""
        self._check_action_shape(action_chunk)
        batch_size = action_chunk.shape[0]
        flat_action = action_chunk.reshape(batch_size, self.action_numel)
        flat_latent = self.encoder(flat_action)
        return flat_latent.reshape(batch_size, self.latent_channels, self.latent_height, self.latent_width)

    def decode_frame(self, action_latent_frame: torch.Tensor) -> torch.Tensor:
        """Decode an action latent frame `(B, C, H, W)` into `(B, chunk_size, action_dim)`."""
        self._check_frame_shape(action_latent_frame)
        batch_size = action_latent_frame.shape[0]
        flat_latent = action_latent_frame.reshape(batch_size, self.latent_numel)
        flat_action = self.decoder(flat_latent)
        return flat_action.reshape(batch_size, self.chunk_size, self.action_dim)

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        return self.decode_frame(self.encode_frame(action_chunk))
