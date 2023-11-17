
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


import copy
from typing import Optional, Tuple, Union, List, Dict
from absl.testing import parameterized
import numpy as np
from gym import spaces
import torch
import unittest
from collections import OrderedDict

BATCH_SIZE = 3
TIME_SEQUENCE_LENGTH = 3
HEIGHT = 256
WIDTH = 320
NUM_IMAGE_TOKENS = 2

# For now, we use one type of spaces.
def space_names_list() -> List[str]:
    """Lists the different types of spaces accepted by the transformer."""
    return ['default']


def state_space_list() -> List[spaces.Dict]:
    """Lists the different types of state spec accepted by the transformer."""
    # This will be input_tensor_spec
    state_space = spaces.Dict(
        {
            'image': spaces.Box(low=0.0, high=1.0, 
                            shape=(3, HEIGHT, WIDTH), dtype=np.float32),
            'natural_language_embedding': spaces.Box(low=-np.inf, high=np.inf, 
                            shape=[512], dtype=np.float32)
        }
    )

    return [state_space]

def observations_list(training: bool = True) -> List[Dict[str, torch.Tensor]]:
    """Lists the different types of observations accepted by the transformer."""
    if training:
        image_shape = [BATCH_SIZE, TIME_SEQUENCE_LENGTH, 3, HEIGHT, WIDTH]
        emb_shape = [BATCH_SIZE, TIME_SEQUENCE_LENGTH, 512]
    else:
        # inference currently only support batch size of 1
        image_shape = [1, 3, HEIGHT, WIDTH]
        emb_shape = [1, 512]
    return [
            {
                'image': torch.full(image_shape, 0.5),
                'natural_language_embedding': torch.full(emb_shape, 1.0)
            }
        ]

NAME_TO_STATE_SPACES = dict(zip(space_names_list(), state_space_list()))
NAME_TO_OBSERVATIONS = dict(zip(space_names_list(), observations_list()))
NAME_TO_INF_OBSERVATIONS = dict(
    zip(space_names_list(), observations_list(False)))

# This class will be inherited by TransformerNetworkTestUtils in transformer_network_test.py.
class TransformerNetworkTestUtils(parameterized.TestCase, unittest.TestCase):
    """Defines spaces, SequenceAgent, and various other testing utilities."""

    def _define_spaces(self,
                train_batch_size=BATCH_SIZE,
                inference_batch_size=1,
                time_sequence_length=TIME_SEQUENCE_LENGTH,
                inference_sequence_length=TIME_SEQUENCE_LENGTH,
                token_embedding_size=512,
                image_width=WIDTH,
                image_height=HEIGHT):
        """Defines spaces and observations (both training and inference)."""

        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.time_sequence_length = time_sequence_length
        self.inference_sequence_length = inference_sequence_length
        self.token_embedding_size = token_embedding_size

        # action space need to keep order of actions so that it can tokenize and detokenize actions correctly.
        # So we define action_space with OrderedDict.
        action_space = spaces.Dict(
            OrderedDict([
                ('terminate_episode', spaces.Discrete(2)), 
                ('world_vector', spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)),
                ('rotation_delta', spaces.Box(low= -np.pi / 2, high= np.pi / 2, shape=(3,), dtype=np.float32)),
                ('gripper_closedness_action', spaces.Box(low= -1.0  , high= 1.0, shape=(1,), dtype=np.float32))
                ])
        )

        state_space = spaces.Dict(
            {
                'image': spaces.Box(low=0.0, high=1.0, 
                                shape=(3, image_height, image_width), dtype=np.float32),
                'natural_language_embedding': spaces.Box(low=-np.inf, high=np.inf, 
                                shape=[self.token_embedding_size], dtype=np.float32)
            }
        )

        self._policy_info_space = {
            'return': spaces.Box(low=0.0, high=1.0, shape=(),dtype=np.float32),
            'discounted_return': spaces.Box(low=0.0, high=1.0, shape=(),dtype=np.float32)
        }
        self._state_space = state_space
        self._action_space = action_space

        self._inference_observation = {
            'image':
                torch.full([self.inference_batch_size, 3, image_height, image_width], 1.0),
            'natural_language_embedding':
                torch.full([self.inference_batch_size, self.token_embedding_size], 1.0),
        }

        self._train_observation = {
            'image':
                torch.full(
                    [self.train_batch_size, self.time_sequence_length, 3, image_height, image_width], 0.5),
            'natural_language_embedding':
                torch.full(
                    [self.train_batch_size, self.time_sequence_length, self.token_embedding_size], 1.)
        }
        self._inference_action = {
            'world_vector':
                torch.full([self.inference_batch_size, 3], 0.5),
            'rotation_delta':
                torch.full([self.inference_batch_size, 3], 0.5),
            'terminate_episode':
                torch.full([self.inference_batch_size], 1),
            'gripper_closedness_action':
                torch.full([self.inference_batch_size, 1], 0.5),
        }

        self._train_action = {
            'world_vector':
                torch.full([self.train_batch_size, self.time_sequence_length, 3], 0.5),
            'rotation_delta':
                torch.full([self.train_batch_size, self.time_sequence_length, 3], 0.5),
            'terminate_episode':
                torch.full([self.train_batch_size, self.time_sequence_length], 1),
            'gripper_closedness_action':
                torch.full([self.train_batch_size, self.time_sequence_length, 1], 0.5),
        }

    def setUp(self):
        self._define_spaces()
        super().setUp()

        