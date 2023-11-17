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
from pytorch_robotics_transformer.tokenizers import image_tokenizer
from absl.testing import parameterized

# Run this command above the robotics_transformer_pytorch directory
# python -m pytorch.robotics_transformer.tokenizers.image_tokenizer_test

class ImageTokenizerTest(parameterized.TestCase, unittest.TestCase):

    @parameterized.named_parameters(
      ('sample_image', 300, False, 8),
      ('sample_image_token_learner', 300, True, 8))
    def testTokenize(self, image_resolution, use_token_learner, num_tokens):
        batch = 1
        seq = 2
        tokenizer = image_tokenizer.RT1ImageTokenizer(
            use_token_learner=use_token_learner, 
            num_tokens=num_tokens)
        image = torch.randn(batch, seq, 3,image_resolution, image_resolution)
        image = torch.clamp(image, min=0, max=1)
        context_vector = torch.rand(batch, seq, 512)
        image_tokens = tokenizer(image, context_vector) # [b, t, num_token, 512] or [b, t, 10 * 10, 512]
        if use_token_learner:
            self.assertEqual(list(image_tokens.shape), [batch, seq, num_tokens, 512])
        else:
            self.assertEqual(list(image_tokens.shape), [batch, seq, 100, 512])

if __name__ == '__main__':
    unittest.main()