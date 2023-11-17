# RT-1: Robotics Transformer, in PyTorch
PyTorch re-implementation of [RT-1](https://github.com/google-research/robotics_transformer). This model is based on 2022 paper ["RT-1: Robotics Transformer for Real-World Control at Scale"](https://arxiv.org/abs/2212.06817).
Although this implementation exactly reproduces the [original code](https://github.com/google-research/robotics_transformer) unlike other PyTorch implementation such as [lucidrains version](https://github.com/lucidrains/robotic-transformer-pytorch), I made some changes for PyTorch compatibility or simplification.

## Features
* Film efficient net based image tokenizer backbone
* Token learner based compression of input tokens
* Transformer for end to end robotic control
* Testing utilities


## Main Changes
- Translated TensorFlow implementation into PyTorch implementation.
- Used spaces of OpenAI gym to define observation and action variables instead of using tensor_spec of TF-Agents.
- Abolished 4-d constraint of some specs in transformer_network.py
- Added useful functions and put those in utils.py.
- Added extra comments.
- Didn't implement the add_summaries function of transformer_network.py which is for tensorboard logging.
- Omitted some variables, functions, classes, and tests including those that are no longer necessary for this repository.

## Note
I didn't implement squence_agent.py and its related codes. Because those codes are for training and inference that use the tf_agents library, so I figure it is not necessary in PyTorch implementation. 
This repository enables you to use RT-1 just like the other Reinforcement learning codes that utilize PyTorch and OpenAI Gym, without the tf_agents library.


## License
This library is licensed under the terms of the Apache license.
