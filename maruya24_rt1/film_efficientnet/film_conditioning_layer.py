# Subject to the terms and conditions of the Apache License, Version 2.0 that the original code follows,
# I have retained the following copyright notice written on it.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can find the original code from here[https://github.com/google-research/robotics_transformer].

import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    def __init__(
        self,
        num_channels: int,
        text_embed_dim: int = 512,
    ):
        super().__init__()
        self._projection_add = nn.Linear(text_embed_dim, num_channels)
        self._projection_mult = nn.Linear(text_embed_dim, num_channels)

        # Note that we initialize with zeros because empirically we have found
        # this works better than initializing with glorot.
        nn.init.constant_(self._projection_add.weight, 0)
        nn.init.constant_(self._projection_add.bias, 0)
        nn.init.constant_(self._projection_mult.weight, 0)
        nn.init.constant_(self._projection_mult.bias, 0)

    # conv_filter: feature maps which corresponds to F in FiLM  paper. (B, C, H, W)
    # conditioning: text which corresponds to x in FiLM paper. this is one vector that is created from a text,
    # note that this is not embedding vectors from a text. Please refer to Universal Sentence Encoder. (B, D). D = 512.
    def forward(self, conv_filters: torch.Tensor, conditioning: torch.Tensor):
        projected_cond_add = self._projection_add(conditioning)  # (B, D) -> (B, C)
        projected_cond_mult = self._projection_mult(conditioning)

        projected_cond_add = projected_cond_add.unsqueeze(2).unsqueeze(
            3
        )  # (B, C) -> (B, C, 1, 1)
        projected_cond_mult = projected_cond_mult.unsqueeze(2).unsqueeze(3)

        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        # see 7.2 section in FiLM paper
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add

        return result
