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


"""Encoder based on Efficientnet."""


# film_efficientnet_encoder receive [bs, 3, 300, 300] and returns [bs, 1536, 10, 10]
# Here we use 1x1 conv. [bs, 1536, 10, 10] -> [bs, 512, 10, 10]
# then apply FiLM.

import torch
import torch.nn as nn
from typing import Optional

try:
    from maruya24_rt1.film_efficientnet.film_efficientnet_encoder import EfficientNetB3
    from maruya24_rt1.film_efficientnet.film_conditioning_layer import FilmConditioning
except:
    from film_efficientnet_encoder import EfficientNetB3
    from film_conditioning_layer import FilmConditioning


class EfficientNetEncoder(nn.Module):
    def __init__(
        self,
        token_embedding_size: int = 512,
        language_embedding_size: int = 512,
        weights: Optional[str] = "imagenet",
        early_film: bool = True,
        include_top: bool = False,
        pooling: bool = True,
    ):
        super().__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels=1536,  # If we use EfficientNetB3 and input image has 3 channels, in_channels is 1536.
            out_channels=token_embedding_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.net = EfficientNetB3(
            weights=weights,
            include_top=include_top,
            include_film=early_film,
            text_embed_dim=language_embedding_size,
        )
        self.film_layer = FilmConditioning(
            num_channels=token_embedding_size, text_embed_dim=language_embedding_size
        )

        self.early_film = early_film
        self._pooling = pooling

    def _encode(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run the image through the efficientnet encoder."""
        if self.early_film:
            return self.net(image, context)
        return self.net(image)

    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = self._encode(image, context)
        features = self.conv1x1(features)
        features = self.film_layer(features, context)

        if not self._pooling:
            return features

        # Global average pool.
        return torch.mean(features, dim=(2, 3))


if __name__ == "__main__":
    enb3 = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        include_film=True,
        text_embed_dim=512,
    ).cuda()
    print(enb3)
    out = enb3(
        torch.randn(torch.Size([40, 3, 256, 320]), device=torch.device("cuda")),
        torch.randn(torch.Size([40, 512]), device=torch.device("cuda")),
    )
    print(out)
