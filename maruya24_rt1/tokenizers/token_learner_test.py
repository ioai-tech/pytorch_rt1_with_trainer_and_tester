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
from pytorch_robotics_transformer.tokenizers.token_learner import TokenLearnerModule

# Type this command on your terminal aboved robotics_transformer_pytorch directory.
# python -m robotics_transformer_pytorch.tokenizer.token_learnerr_test

class TokenLearnerTest(unittest.TestCase):
    def testTokenLearner(self, embedding_dim=512, num_tokens=8):
        batch = 1
        seq = 2
        token_learner_layer = TokenLearnerModule(
            inputs_channels=embedding_dim, 
            num_tokens=num_tokens)
        # seq is time-series length
        # embedding_dim = channels
        inputvec = torch.randn((batch * seq, embedding_dim, 10, 10))

        learnedtokens = token_learner_layer(inputvec)
        self.assertEqual(list(learnedtokens.shape), [batch * seq, num_tokens, embedding_dim])

if __name__ == '__main__':
    unittest.main()