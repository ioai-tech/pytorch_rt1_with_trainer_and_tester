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

from maruya24_rt1.tokenizers import action_tokenizer
from maruya24_rt1.tokenizers import image_tokenizer
from maruya24_rt1.transformer import Transformer
from maruya24_rt1.film_efficientnet import preprocessors

from typing import Optional, Tuple, Union, Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F


# This is a robotics transformer network.
class TransformerNetwork(nn.Module):
    """A transformer based actor network."""

    def __init__(
        self,
        input_tensor_space: spaces.Dict,  # observatioin space like dict. keys are image, natural_language_embedding
        output_tensor_space: spaces.Dict,  # action space like dict. keys are world_vector, rotation_delta, gripper_closedness_action, terminate_episode
        train_step_counter: int = 0,
        vocab_size: int = 256,  # Dimensionality of tokens from the output layer. This is also dimensionality of tokens from the input layer.
        token_embedding_size: int = 512,  # RT1ImageTokenizer outputs(=context_image_tokens) has embedding dimension of token_embedding_size. This will finally be utilized in 1x1 Conv in EfficientNetEncoder class.
        language_embedding_size: int = 512,  # embedding size for language embedding and Film layers
        num_layers: int = 1,
        layer_size: int = 4096,  # This corresponds to key_dim which is the size of each attention head for query, key and values.
        num_heads: int = 8,
        feed_forward_size: int = 512,  # This corresponds to d_model which is embedding dimension of each token in transformer part.
        dropout_rate: float = 0.1,
        time_sequence_length: int = 1,
        crop_size: int = 236,
        # action_order: Optional[List[str]] = None,
        use_token_learner: Optional[bool] = True,
        return_attention_scores: bool = False,
        num_encoders=1,
        using_proprioception=False,
    ):
        super().__init__()

        self._input_tensor_space = input_tensor_space
        self._output_tensor_space = output_tensor_space
        self._train_step_counter = train_step_counter
        self._actions = None
        self._returns = None
        self._vocab_size = vocab_size
        self._token_embedding_size = token_embedding_size
        self._language_embedding_size = language_embedding_size
        self._time_sequence_length = time_sequence_length
        self._crop_size = crop_size
        self.num_encoders = num_encoders
        # creatr transformer

        # create tokenizers
        self._image_tokenizers = nn.ModuleDict()
        self.language_embedding = nn.Embedding(512, language_embedding_size)
        for idx_encoder in range(num_encoders):
            self._image_tokenizers[str(idx_encoder)] = (
                image_tokenizer.RT1ImageTokenizer(
                    embedding_output_dim=self._token_embedding_size,
                    language_embedding_size=self._language_embedding_size,
                    use_token_learner=use_token_learner,
                    num_tokens=8,
                )
            )
            self._image_tokenizers[str(idx_encoder)].eval()
        # self._image_tokenizer = image_tokenizer.RT1ImageTokenizer(
        #     embedding_output_dim=self._token_embedding_size,
        #     use_token_learner=use_token_learner,
        #     num_tokens=8)
        self._action_tokenizer = action_tokenizer.RT1ActionTokenizer(
            output_tensor_space, vocab_size=self._vocab_size  # action space
        )
        self.using_proprioception = using_proprioception
        if self.using_proprioception:
            self._transformer = Transformer(
                num_layers=num_layers,
                layer_size=layer_size,
                num_heads=num_heads,
                feed_forward_size=feed_forward_size,
                dropout_rate=dropout_rate,
                vocab_size=self._vocab_size,
                input_token_emb_dim=self._token_embedding_size + 1,
                return_attention_scores=return_attention_scores,
            )
        else:
            self._transformer = Transformer(
                num_layers=num_layers,
                layer_size=layer_size,
                num_heads=num_heads,
                feed_forward_size=feed_forward_size,
                dropout_rate=dropout_rate,
                vocab_size=self._vocab_size,
                input_token_emb_dim=self._token_embedding_size,
                return_attention_scores=return_attention_scores,
            )

        """
        following the model design in diffusion policy, we concatenate robot ee's position and orientation at the end of sequence
        """

        # Get the number of tokens
        self._tokens_per_action = self._action_tokenizer.tokens_per_action
        self._tokens_per_context_image = self._image_tokenizers[
            "0"
        ].tokens_per_context_image

        # generate loss mask and attention mask
        self._generate_masks()

        # define mappings to token embedding size
        # self._action_token_emb = nn.Linear(self._vocab_size, self._token_embedding_size)

        # define loss function
        self.loss_weight = torch.ones(vocab_size)
        self._loss_object = nn.CrossEntropyLoss(
            reduction="mean", weight=self.loss_weight
        )

        self._attention_scores = []
        self._use_token_learner = use_token_learner

        # this is used only when random sampling
        # when sampling, the output is used as network_state
        self._state_space = spaces.Dict(
            {
                "context_image_tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        time_sequence_length,
                        self._tokens_per_context_image,
                        token_embedding_size,
                    ),
                    dtype=np.float32,
                ),
                "context_pos_orn": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(time_sequence_length, self._tokens_per_context_image),
                    dtype=np.float32,
                ),
                "action_tokens": spaces.MultiDiscrete(
                    np.full((time_sequence_length, self._tokens_per_action), vocab_size)
                ),
                # Stores where in the window we are.
                # This value is within range [0, time_sequence_length + 1].
                # When seq_idx == time_sequence_length, context_image_tokens and
                # action_tokens need to be shifted to the left.
                "seq_idx": spaces.Discrete(time_sequence_length + 1),
                # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                # 1 time step means [context_image_tokens + action_tokens]
                # seq_idx means which time steps we are. But it is adjusted to time_sequence_length when it exceeds time_sequence_length.
            }
        )

    @property
    def attention_scores(self) -> List[torch.Tensor]:
        """Return attention score. This is for debugging/visualization purpose."""
        return self._attention_scores

    def _get_action_index_for_token(self, k):
        """Returns action associated with the token at given position `k`.

        If k is not an action token then it returns -1.
        If k is part of the first action in the sequence then returns 0 etc.

        Args:
            k: an int that represents the position in the sequence.

        Returns:
            The index of the action that this position belongs to, or if this
            position is part of an image token then returns -1.
        """
        if k < 0 or k >= self._all_num_tokens:
            return -1

        n = k
        if (
            n % self._single_time_step_num_tokens < self._tokens_per_context_image
        ):  # check whether k is context_image token
            return -1
        return int(
            n / self._single_time_step_num_tokens
        )  # return which time index that k belongs to.

    # _action_tokens_mask is for loss computing. This has all indexes of action tokens in all tokens.
    # We can know which output tokens are action predictions by _action_tokens_mask - 1.
    # _default_attention_mask is modified causaul mask because we will only use observations tokens when predicting actions.
    # So we also have to mask action tokens.
    def _generate_masks(self):
        """Generate mask for action prediction loss and attention visualization."""
        # each time step = [image, action]
        self._single_time_step_num_tokens = (
            self._tokens_per_action + self._tokens_per_context_image
        )

        # full sequence = [prefix context + N x timestep + postfix context]
        self._all_num_tokens = (
            self._time_sequence_length * self._single_time_step_num_tokens
        )

        # create mask for action predition loss
        # self._action_tokens_mask has all indexes of action tokens in all tokens.
        self._action_tokens_mask = []
        for n in range(0, self._all_num_tokens, self._single_time_step_num_tokens):
            for x in range(0, self._tokens_per_action, 1):
                self._action_tokens_mask.append(x + n + self._tokens_per_context_image)

        # The look ahead mask ensures causality.
        # This is a lower triangular matrix. All elements other than 0 are 1.
        # 0 means mask.
        self._default_attention_mask = torch.tril(
            torch.ones((self._all_num_tokens, self._all_num_tokens), dtype=torch.uint8)
        )

        action_mask = torch.from_numpy(
            np.ndarray(shape=(self._all_num_tokens, self._all_num_tokens), dtype=int)
        )

        for i in range(self._all_num_tokens):
            for j in range(self._all_num_tokens):
                action_i = self._get_action_index_for_token(i)
                action_j = self._get_action_index_for_token(j)
                mask = 0
                if (
                    action_i != -1 and action_j != -1
                ):  # Check both of i and j are actions.
                    # Ignore actions of previous time steps.
                    if action_j < action_i:
                        mask = 1
                    # If we're not auto-regression, ignore action of current time step.
                    if action_j == action_i and j <= i:
                        mask = 1
                action_mask[i, j] = mask
        action_mask = action_mask.to(self._default_attention_mask.device)
        self._default_attention_mask -= action_mask

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        network_state: Dict[
            str, torch.Tensor
        ],  # network_state retain obvervation toknes, action tokens, seq_idx.
    ):
        """Calls the transformer network.

        Args:
            observations: Observation data including image and natural language
                embedding in dict of Tensors.
        network_state: Network state data including time step, image, action
            tokens, step number in dict of Tensors.

        Returns:
            A tuple `(Detokenized output actions, network state)`.
        """

        # used to determine training vs inference call
        # outer_rank will be 2 -> [b, t] during training and
        # outer_rank will be 1 -> [b] during inference
        # print('observations: ')
        # for k, v in observations.items():
        #     print(k + ' : ' + str(v.size()))
        # print('network_state: ')
        # for k, v in network_state.items():
        #     print(k + ' : ' + str(v.size()))
        outer_rank = self._get_outer_rank(observations)
        assert outer_rank in (1, 2), "outer rank should be 1 or 2"

        b, t = self._get_batch_size_and_seq_len(network_state)
        # network_state is used when inference.
        # b : batch size
        # t: time_sequence_length of this model

        # context_image_tokens: (b, t, num_tokens, embedding_dim)
        # action_tokens: (b, t, self._tokens_per_action)
        context_image_tokens, action_tokens, attention_mask = self._get_tokens_and_mask(
            observations, network_state
        )

        self._aux_info = {"action_labels": action_tokens}

        if outer_rank == 1:  # This is an inference call
            # run transformer in loop to produce action tokens one-by-one
            seq_idx = network_state["seq_idx"][0]
            action_t = torch.minimum(
                seq_idx, torch.tensor(self._time_sequence_length - 1)
            )
            # Transformer shifts all to the left by one step by default (it's usually
            # predicting the next token as default training task...).
            transformer_shift = -1
            # We only want to get the action predicted at time_step.
            # This index means the output of the last observation token that is at action_t time step.
            start_index = (
                transformer_shift
                + self._tokens_per_context_image
                + action_t * (self._single_time_step_num_tokens)
            )
            current_action_tokens = []
            action_predictions_logits = []
            # Repeat inference tokens_per_action times.
            for k in range(self._tokens_per_action):
                action_index = start_index + k
                # token: (1, 1)
                # token_logits: (1, 1 vocab_size)
                token, token_logits = self._transformer_call_and_slice(
                    context_image_tokens,
                    action_tokens,
                    attention_mask=attention_mask,
                    batch_size=b,
                    slice_start=action_index,  # slicing single action dimension
                )
                action_predictions_logits.append(token_logits)

                current_action_tokens.append(token)

                # Add the predicted token to action_tokens
                action_tokens = action_tokens.view(
                    b, -1
                )  # [b, t, self._tokens_per_action] -> [b, t * self._tokens_per_action]
                action_start_index = (action_t * self._tokens_per_action) + k
                # replace action_tokens[:, action_start_index] with the predicted token. Note that this is not insert.
                action_tokens = torch.concat(
                    [
                        action_tokens[:, :action_start_index],
                        token,
                        action_tokens[:, action_start_index + 1 :],
                    ],
                    dim=1,
                )
                action_tokens = action_tokens.view(
                    b, t, self._tokens_per_action
                )  # [b, t * self._tokens_per_action] -> [b, t, self._tokens_per_action]

            self._aux_info.update(
                {
                    # action_predictions_logits is
                    # [1, self._tokens_per_action, self._vocab_size]
                    "action_predictions_logits": torch.concat(
                        action_predictions_logits, 1
                    )
                }
            )

            predicted_tokens_for_output = torch.concat(
                current_action_tokens, 1
            )  # [1, self._tokens_per_action]
            one_state_action_tokens = predicted_tokens_for_output.unsqueeze(
                1
            )  # [1, 1, self._tokens_per_action]

            # Add predicted action tokens  to network_state['action_tokens']
            state_action_tokens = network_state[
                "action_tokens"
            ]  # (1, time_sequence_length, self._tokens_per_action)
            # replace state_action_tokens[:, action_t, ...] with the predicted tokens. Note that this is not insert.
            network_state["action_tokens"] = torch.concat(
                [
                    state_action_tokens[:, :action_t, ...],
                    one_state_action_tokens,
                    state_action_tokens[:, action_t + 1 :, ...],
                ],
                dim=1,
            )

            # Increment the time_step for the next inference call.
            # network_state['seq_idx'] never exceed time_sequence_length.
            network_state["seq_idx"] = torch.minimum(
                seq_idx + 1, torch.tensor(self._time_sequence_length)
            )[None]

            self._loss = torch.tensor(0.0)

        else:
            # training call --> simply run one transformer forward pass
            # output_tokens: (bs, t*num_tokens, vocab_size)
            output_tokens = self._transformer_call(
                context_image_tokens,
                action_tokens,
                attention_mask=attention_mask,
                batch_size=b,
            )

            # Gather all predicted actions for the action loss. Use fancy index to extract all predicted actions.
            predicted_action_index = torch.tensor(self._action_tokens_mask) - 1
            action_logits = output_tokens[
                :, predicted_action_index
            ]  # (bs, t*tokens_per_action, vocab_size)
            action_logits_for_training = action_logits.view(
                b, t, self._tokens_per_action, -1
            )  # (bs, t, self._tokens_per_action, vocab_size)

            # Only take the last action as the action.
            # action_logits_for_output is [b, self._tokens_per_action, emb]
            action_logits_for_output = action_logits_for_training[
                :, -1
            ]  # This will take action at last time step in this training.
            # predicted_tokens_for_output is [b, self._tokens_per_action]
            predicted_tokens_for_output = torch.argmax(action_logits_for_output, dim=-1)
            action_logits_for_training = action_logits_for_training.permute(0, 3, 1, 2)
            num_items = float(b * t) * self._single_time_step_num_tokens
            # action_logits_for_training: (b, t, self._tokens_per_action, vocab_size)
            # action_tokens, (b, t, self._tokens_per_action)
            # action_loss: (b, t)
            action_loss = self._loss_object(
                action_logits_for_training, action_tokens
            )  # (b, t, self._tokens_per_action)

            self._loss = action_loss

            # store action labels and predictions for visualization
            self._aux_info.update(
                {
                    "action_predictions": torch.argmax(
                        action_logits_for_training, dim=-1
                    ),
                    "action_loss": action_loss,
                    "actor_loss_mask": torch.ones((b), dtype=torch.float32),
                }
            )

        # output_actions: Dict[str, np.ndarray]
        output_actions = self._action_tokenizer.detokenize(predicted_tokens_for_output)

        # output_actions is the last actions.
        # network_stape is the past state that is used for next inference.
        return output_actions, network_state

    def _get_outer_rank(self, observations: Dict[str, torch.Tensor]) -> int:
        # used to determine training vs inference call
        # outer_rank will be 2 -> [b, t] during training and
        # outer_rank will be 1 -> [b] during inference

        for k in observations.keys():
            obs_value = observations[k]
            obs_value_shape = obs_value.shape

            obs_space = self._input_tensor_space[k]
            obs_space_shape = obs_space.shape
            break
        return len(obs_value_shape) - len(obs_space_shape)

    def _get_batch_size_and_seq_len(self, network_state):
        image_shape = network_state["context_image_tokens"].shape
        b = image_shape[0]
        t = image_shape[1]
        return b, t

    def _transformer_call(
        self,
        context_image_tokens: torch.Tensor,  # (b, t, num token, emb_dim)
        action_tokens: torch.Tensor,  # (b, t, self._tokens_per_action)
        batch_size: int,
        attention_mask: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_token_sequence = self._assemble_input_token_sequence(
            context_image_tokens, action_tokens, batch_size
        )  # [b, t*num_tokens, emb_dim]
        # run transformer
        output_tokens, self._attention_scores = self._transformer(
            input_token_sequence, attention_mask
        )  # (bs, t*num_tokens, vocab_size)
        return output_tokens

    # input_token_sequence = [context_image_tokens + action_tokens]
    def _assemble_input_token_sequence(
        self, context_image_tokens, action_tokens, batch_size
    ):
        # embed action tokens
        # action_tokens = F.one_hot(action_tokens, num_classes=self._vocab_size).to(torch.float32)
        # action_tokens = self._action_token_emb(action_tokens) # [b, t , num_action_tokens, emb_dim]

        action_tokens = torch.zeros_like(
            context_image_tokens
        )  # This removes autoregressively conditioning on actions becuase it did not benefit performance and slowed inference.
        # assemble token sequence
        input_token_sequence = torch.concat(
            (context_image_tokens, action_tokens), dim=2
        )
        if self.using_proprioception:
            input_token_sequence = input_token_sequence.view(
                batch_size, -1, self._token_embedding_size + 1
            )
        else:
            input_token_sequence = input_token_sequence.view(
                batch_size, -1, self._token_embedding_size
            )  # [b, t*num_tokens, emb_dim]
        return input_token_sequence

    # Call transformer, slice output, return predicted token.
    def _transformer_call_and_slice(
        self, *args, slice_start: int = 0, slice_length: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = self._transformer_call(*args, **kwargs)

        slice_end = slice_start + slice_length
        token_logits = output_tokens[
            :, slice_start:slice_end, :
        ]  # (b, slice_length, vocab_size)
        token = torch.argmax(token_logits, dim=-1)

        return token, token_logits

    def _get_tokens_and_mask(
        self,
        observations: Dict[str, torch.Tensor],
        network_state: Dict[str, torch.Tensor],
    ):
        # tokenize all inputs
        context_image_tokens, network_state = self._tokenize_images(
            observations, network_state
        )
        # print(f"size of context_image_tokens in _get_tokens_and_mask: {context_image_tokens.size()}")
        # print(f"size of network_state in _get_tokens_and_mask: {network_state.size()}")
        # if self.using_proprioception:

        # else:
        #     position_and_orn = torch.zeros_like(context_image_tokens)
        action_tokens = self._tokenize_actions(observations, network_state)

        # generate transformer attention mask
        attention_mask = self._default_attention_mask

        return (context_image_tokens, action_tokens, attention_mask)

    # At training, we don't use network_state at all.
    # At training, this will just convert image and context into tokens.
    def _tokenize_images(self, observations, network_state):
        image = observations["image"]  # [b, t, c, h, w] or [b, c, h, w]
        outer_rank = self._get_outer_rank(observations)

        if outer_rank == 1:  # This is an inference call
            seq_idx = network_state["seq_idx"][0]  # 0 ~ time_sequence_length
            time_step = torch.minimum(
                seq_idx, torch.tensor(self._time_sequence_length - 1)
            )
            image = image.unsqueeze(1)  # [b, c, h, w] -> [b, 1, c, h, w]

        image_shape = image.shape
        b = image_shape[0]
        input_t = image_shape[1]
        c = image_shape[2]
        h = image_shape[3]
        w = image_shape[4]

        # return context from observation after check whether context is in observation.
        context = self._extract_context_from_observation(
            observations, input_t
        )  # [b, t, emb-size] or None

        # preprocess image
        image = image.view(
            (b * input_t, c, h, w)
        )  # image is already tensor and its range is [0,1]
        # print(f"size of image in _tokenize_images before preprocess: {image.size()}")

        # image = preprocessors.convert_dtype_and_crop_images(image)
        image = image.view((b, input_t, c, h, w))
        # print(f"size of image in _tokenize_images before preprocess: {image.size()}")

        # context_image_tokens = torch.zeros(
        #     size=(
        #         self.num_encoders,
        #         b,
        #         self._time_sequence_length,
        #         8,
        #         self._token_embedding_size,
        #     ),
        #     dtype=torch.float32,
        #     device=torch.device("cuda"),
        # )
        context_image_tokens = []
        # get image tokens
        for i in range(self.num_encoders):
            img = image[:, :, :, i * 256 : (i + 1) * 256, :]
            with torch.no_grad():
                context_image_tokens.append(
                    self._image_tokenizers[str(i)](img, context=context)
                )  # (batch, t, num_tokens, embedding_dim)
                # torch.cuda.empty_cache()
        context_image_tokens = sum(context_image_tokens)
        num_tokens = context_image_tokens.shape[2]
        if self.using_proprioception:
            context_pos_orn = torch.cat(
                (observations["position"], observations["orientation"]), dim=-1
            )
            context_pos_orn = context_pos_orn.view(b, input_t, context_pos_orn.size(-1))
            # context_pos_orn = self.pos_orn_encoder(context_pos_orn.unsqueeze(-1))
        # update network state at inference
        # At inference, we retain some context_image_tokens to accelerate computation.
        # At inference, context_image_tokens : (batch, 1, num_tokens, embedding_dim)
        # At inference, network_state stores context_image_tokens of past time steps.
        # Here, we combine past context_image_tokens of network_state with current context_image_tokens.
        # network_state only store tokens within time_sequence_length time steps.
        # This means network_state does not store the tokens for all past steps, but only for time_sequence_length time steps.
        # if current time step >= time_sequence_length, we store context_image_tokens after we discard the oldest context_image_tokens.
        # Here, we implement that by shifting state_image_token to the left.
        if outer_rank == 1:  # This is an inference call
            state_image_tokens = network_state[
                "context_image_tokens"
            ]  # (1, time_sequence_length, tokens_per_context_image, token_embedding_size)
            # network_state as input for this call is the output from the last call.
            # Therefore, we need to shift all images to the left by 1 in the time axis
            # to align with the time dim in this call.
            state_image_tokens = (
                torch.roll(state_image_tokens, -1, 1)
                if seq_idx == self._time_sequence_length
                else state_image_tokens
            )
            # if seq_idx == time_sequence_length, state_image_tokens will be shifted to the left along time axis
            # seq_idx will be incremented in forward function. But it is adjusted so that it never exceed time_sequence_length.
            # Therefore, shiftimg will always occur when time step exceeds time_sequence_length.
            # maximum of time_step is self._time_sequence_length - 1
            context_image_tokens = torch.concat(
                [
                    state_image_tokens[:, :time_step, ...],
                    context_image_tokens,  # Note that in inference, size of context_image_tokens is (batch, 1, num_tokens, embedding_dim)
                    state_image_tokens[
                        :, time_step + 1 :, ...
                    ],  # if time_step == time_sequence_lengths -1, this will be empty tensor.
                    # So this tensor will be ignored when concat
                ],
                dim=1,
            )
            network_state["context_image_tokens"] = context_image_tokens
            if self.using_proprioception:
                state_pos_orn = network_state["context_pos_orn"]
                state_pos_orn = (
                    torch.roll(state_pos_orn, -1, 1)
                    if seq_idx == self._time_sequence_length
                    else state_pos_orn
                )
                context_pos_orn = torch.concat(
                    [
                        state_pos_orn[:, :time_step, ...],
                        context_pos_orn,  # Note that in inference, size of context_image_tokens is (batch, 1, num_tokens, embedding_dim)
                        state_pos_orn[
                            :, time_step + 1 :, ...
                        ],  # if time_step == time_sequence_lengths -1, this will be empty tensor.
                        # So this tensor will be ignored when concat
                    ],
                    dim=1,
                )
                network_state["context_pos_orn"] = context_pos_orn
        if self.using_proprioception:
            context_image_tokens = torch.cat(
                (context_image_tokens, context_pos_orn.unsqueeze(-1)), dim=-1
            )
        # context_image_tokens = context_image_tokens + context_pos_orn if self.using_proprioception else context_image_tokens

        return context_image_tokens, network_state

    def _tokenize_actions(self, observations, network_state):
        outer_rank = self._get_outer_rank(observations)

        if outer_rank == 1:  # This is an inference call
            action_tokens = network_state["action_tokens"]
            seq_idx = network_state["seq_idx"][0]
            # network_state as input for this call is the output from the last call.
            # Therefore, we need to shift all actions by 1 to the left.
            action_tokens = (
                torch.roll(action_tokens, -1, 1)
                if seq_idx == self._time_sequence_length
                else action_tokens
            )
        else:
            assert outer_rank == 2
            # self._actions was set through set_actions function.
            if (
                self._actions is None
            ):  # When there is no action that will be tokenized to begin with, we create zero tensor.
                b, t = self._get_batch_size_and_seq_len(network_state)
                action_tokens = torch.zeros(
                    shape=[b, t, self._tokens_per_action], dtype=torch.int32
                )
            else:
                action_tokens = self._action_tokenizer.tokenize(self._actions)
        return action_tokens

    # output context from observation. size: [b, t, emb-size]
    def _extract_context_from_observation(self, observations, seq_len):
        """Extract context from observation."""
        context = None
        if "natural_language_embedding" in observations:
            outer_rank = self._get_outer_rank(observations)

            context = self.language_embedding(
                observations["natural_language_embedding"]
            )  # [b, t, emb-size] or [b, emb-size]
            if outer_rank == 1:
                context = torch.tile(context[:, None], [1, seq_len, 1])
                # [b, emb-size] ->  [b, 1, emb-size] -> [b, seq_len, emb-size]
        return context

    # Actions is passed to this class only through this function.
    def set_actions(self, actions: Dict[str, torch.Tensor]):
        """Sets actions that will be tokenized and used in transformer network.

        Args:
        actions: actions to be tokenized and used in transformer network. example
            actions are terminate = 1 world_vector = [0.9, 0.8, -0.3]
            rotation_delta = [-0.1, 0.2, .6] gripper_closedness = 0.9
        ※ terminate is either 0 or 1.
        """
        self._actions = actions

    def get_actor_loss(self) -> torch.Tensor:
        return self._loss

    def get_aux_info(self) -> Dict[str, Any]:
        return self._aux_info


if __name__ == "__main__":
    net = TransformerNetwork()
