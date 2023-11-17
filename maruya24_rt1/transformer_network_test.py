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


"""Tests for networks."""

import torch
import torch.nn.functional as F
from absl.testing import parameterized
import unittest
import numpy as np
from typing import Dict
import sys
sys.path.append("/home/io011/workspace/robotics_transformer/")
import transformer_network
from transformer_network_test_set_up import BATCH_SIZE
from transformer_network_test_set_up import NAME_TO_INF_OBSERVATIONS
from transformer_network_test_set_up import NAME_TO_STATE_SPACES
from transformer_network_test_set_up import observations_list
from transformer_network_test_set_up import space_names_list
from transformer_network_test_set_up import state_space_list
from transformer_network_test_set_up import TIME_SEQUENCE_LENGTH
from transformer_network_test_set_up import TransformerNetworkTestUtils
from tokenizers.utils import batched_space_sampler
from tokenizers.utils import np_to_tensor


class TransformerNetworkTest(TransformerNetworkTestUtils):
    @parameterized.named_parameters([{
        'testcase_name': '_' + name,
        'state_space': spec,
        'train_observation': obs,
    } for (name, spec, obs) in zip(space_names_list(), state_space_list(), observations_list())])
    def testTransformerTrainLossCall(self, state_space, train_observation):
        network = transformer_network.TransformerNetwork(
        vocab_size=256,
        token_embedding_size=512,
        num_layers=8,
        layer_size=128,
        num_heads=8,
        feed_forward_size=512,
        dropout_rate=0.1,
        crop_size=236,
        input_tensor_space=state_space,
        output_tensor_space=self._action_space,
        time_sequence_length=TIME_SEQUENCE_LENGTH,
        use_token_learner=True)


        network.set_actions(self._train_action)

        network_state = batched_space_sampler(network._state_space, batch_size=BATCH_SIZE)
        network_state = np_to_tensor(network_state) # change np.ndarray type of sample values into tensor type

        output_actions, network_state = network(
            train_observation, network_state=network_state)

        expected_shape = [2, 3]

        self.assertEqual(list(network.get_actor_loss().shape), expected_shape)

        self.assertCountEqual(self._train_action.keys(), output_actions.keys())

    @parameterized.named_parameters([{
        'testcase_name': '_' + name,
        'space_name': name,
    } for name in space_names_list()])
    def testTransformerInferenceLossCall(self, space_name):
        state_space = NAME_TO_STATE_SPACES[space_name]
        observation = NAME_TO_INF_OBSERVATIONS[space_name] #  observation has no time dimension unlike during training.

        network = transformer_network.TransformerNetwork(
        input_tensor_space=state_space,
        #Dict('image': Box(0.0, 1.0, (3, 256, 320), float32), 'natural_language_embedding': Box(-inf, inf, (512,), float32))
        output_tensor_space=self._action_space,
        # Dict('terminate_episode': Discrete(2), 'world_vector': Box(-1.0, 1.0, (3,), float32), 'rotation_delta': Box(-1.5707964, 1.5707964, (3,), float32), 'gripper_closedness_action': Box(-1.0, 1.0, (1,), float32))
        time_sequence_length=TIME_SEQUENCE_LENGTH)

        network.set_actions(self._inference_action) # self._inference_action has no time dimension unlike self._train_action.
        # inference currently only support batch size of 1
        network_state = batched_space_sampler(network._state_space, batch_size=1)
        network_state = np_to_tensor(network_state) # change np.ndarray type of sample values into tensor type

        output_actions, network_state = network(
            observation, network_state=network_state)
        # torch.float32 torch.Size([1, 3, 256, 320])
        # {'image': tensor([[[[0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   ...,
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000]],

        #  [[0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   ...,
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000]],

        #  [[0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   ...,
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000],
        #   [0.5000, 0.5000, 0.5000,  ..., 0.5000, 0.5000, 0.5000]]]]), 
        # 
        # torch.float32  torch.Size([1, 512])
        # 'natural_language_embedding': tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        #  1., 1., 1., 1., 1., 1., 1., 1.]])}

       # torch.int64 torch.Size([1, 3, 8])
    #     {'action_tokens': tensor([[[116,  61,  70, 126, 170, 243,  58, 127],
    #      [167, 251, 204,  19, 222, 236,  61,  91],
    #      [254, 249, 204,  14,  64, 162,  53, 248]]]), 
    # torch.float32 torch.Size([1, 3, 8, 512])
    # 'context_image_tokens': tensor([[[[-0.8647, -1.9947, -0.7335,  ..., -1.0252, -0.1364,  0.6359],
    #       [ 0.1899,  0.1604, -0.4497,  ..., -0.3601,  0.0933, -0.6361],
    #       [ 1.5404,  0.2238, -1.0574,  ...,  0.0621,  1.2833, -2.0537],
    #       ...,
    #       [ 1.5256, -0.4647, -0.5676,  ...,  1.4825, -0.5317,  0.7946],
    #       [-1.3059,  0.8747, -2.9062,  ..., -0.0762,  1.8340, -0.8561],
    #       [-0.6595, -2.0878,  0.2709,  ...,  0.1764,  1.2843, -0.4609]],

    #      [[-0.0947,  0.0133, -0.2301,  ...,  0.1116, -0.1111, -0.0391],
    #       [-0.0998,  0.0090, -0.2151,  ...,  0.1465, -0.1033, -0.0342],
    #       [-0.0935,  0.0161, -0.2835,  ...,  0.1232, -0.1025, -0.0332],
    #       ...,
    #       [-0.1249, -0.0076, -0.2253,  ...,  0.1392, -0.1412, -0.0528],
    #       [-0.1013, -0.0141, -0.2092,  ...,  0.1403, -0.1279, -0.0494],
    #       [-0.0929,  0.0040, -0.1969,  ...,  0.1384, -0.1017, -0.0603]],

    #      [[-1.8412, -1.0423, -0.4420,  ...,  0.8732, -0.1140, -0.1008],
    #       [ 0.6039, -0.6286, -0.1894,  ..., -0.4552,  1.3157, -1.1879],
    #       [ 1.1295, -0.1596, -0.0862,  ..., -1.0804,  1.0614, -0.0249],
    #       ...,
    #       [-0.3196, -1.3627,  0.3153,  ..., -0.7006,  1.0846, -0.2101],
    #       [ 2.3243,  1.2046,  1.8313,  ..., -1.1140,  0.4493, -0.8400],
    #       [ 1.1736, -0.3540, -0.6332,  ...,  0.0968,  1.5409,  0.7139]]]],
    #    grad_fn=<CatBackward0>), 
    # torch.int64 torch.Size([])
    # 'seq_idx': tensor(2)}

        self.assertEqual(network.get_actor_loss().item(), 0.0)
        self.assertCountEqual(self._inference_action.keys(), output_actions.keys())

    @parameterized.named_parameters([{
      'testcase_name': '_' + name,
      'state_space': spec,
    } for name, spec in zip(space_names_list(), state_space_list())])
    def testTransformerCausality(self, state_space):
        network = transformer_network.TransformerNetwork(
        input_tensor_space=state_space,
        output_tensor_space=self._action_space,
        time_sequence_length=TIME_SEQUENCE_LENGTH,
        dropout_rate=0.0)

        network.eval()

        time_sequence_length = network._time_sequence_length
        tokens_per_image = network._tokens_per_context_image
        tokens_per_action = network._tokens_per_action

        # size of all_tokens: (time_sequence_length * (tokens_per_image + tokens_per_action)) 
        def _split_image_and_action_tokens(all_tokens):
            image_start_indices = [(tokens_per_image + tokens_per_action) * k
                             for k in range(time_sequence_length)]

            image_tokens = torch.stack(
                [all_tokens[i:i + tokens_per_image] for i in image_start_indices],
                dim=0)
            action_start_indices = [i + tokens_per_image for i in image_start_indices]
            action_tokens = torch.stack([
                    all_tokens[i:i + tokens_per_action] for i in action_start_indices],
                    dim=0)

            image_tokens = F.one_hot(image_tokens, network._token_embedding_size)
            # Add batch dimension.
            image_tokens = image_tokens.unsqueeze(0) # image_tokens: (1, time_sequence_length, tokens_per_image, emb_dim)
            action_tokens = action_tokens.unsqueeze(0) # action: (1, time_sequence_length, tokens_per_action)

            return image_tokens, action_tokens

        # Generate some random tokens for image and actions.
        all_tokens = torch.randint(low=0, high=10, size=(time_sequence_length * (tokens_per_image + tokens_per_action),))
        context_image_tokens, action_tokens = _split_image_and_action_tokens(all_tokens)
        # Get the output tokens without any zeroed out input tokens.
        # output_tokens: (t*num_tokens, vocab_size)
        output_tokens = network._transformer_call(
            context_image_tokens=context_image_tokens,
            action_tokens=action_tokens,
            attention_mask=network._default_attention_mask,
            batch_size=1)[0]

        for t in range(time_sequence_length *
                   (tokens_per_image + tokens_per_action)):
            # Zero out future input tokens.
            all_tokens_at_t = torch.concat([all_tokens[:t + 1], torch.zeros_like(all_tokens[t + 1:])], 0)
            context_image_tokens, action_tokens = _split_image_and_action_tokens(all_tokens_at_t)

            # Get the output tokens with zeroed out input tokens after t.
            output_tokens_at_t = network._transformer_call(
                context_image_tokens=context_image_tokens,
                action_tokens=action_tokens,
                attention_mask=network._default_attention_mask,
                batch_size=1)[0]

            # The output token is unchanged if future input tokens are zeroed out.
            np.testing.assert_array_equal(output_tokens[:t + 1].detach().numpy(), output_tokens_at_t[:t + 1].detach().numpy())


if __name__ == '__main__':
    unittest.main()
