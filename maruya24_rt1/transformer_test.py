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

import unittest
import torch

from maruya24_rt1.transformer import Transformer
from absl.testing import parameterized

class TransformerTest(parameterized.TestCase, unittest.TestCase):

    def setUp(self):
        self._vocab_size = 10
        batch_size = 8
        sequence_len = 12
        self._tokens = torch.rand((batch_size, sequence_len, self._vocab_size))

    @parameterized.parameters(True, False)
    def test_transformer_forwardpass(self, return_attention_scores):
        network = Transformer(
            num_layers=2,
            layer_size=512,
            num_heads=4,
            feed_forward_size=256,
            dropout_rate=0.1,
            vocab_size=self._vocab_size,
            input_token_emb_dim=self._vocab_size,
            return_attention_scores=return_attention_scores,
            max_seq_len=15)

        output_tokens, attention_scores = network(self._tokens, attention_mask=None)
        self.assertSequenceEqual(self._tokens.shape, output_tokens.shape)
        if return_attention_scores:
            self.assertNotEmpty(attention_scores)
        else:
            self.assertEmpty(attention_scores)



if __name__ == '__main__':
    unittest.main()