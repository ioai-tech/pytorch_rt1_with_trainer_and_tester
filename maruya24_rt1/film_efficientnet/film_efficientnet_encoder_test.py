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
from skimage import data
from absl.testing import parameterized
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_robotics_transformer.film_efficientnet.film_efficientnet_encoder import EfficientNetB3, ILSVRCPredictor


# If you want run this test, move to the directory above pytorch_robotics_transformer directory, type below command on terminal.
# python -m pytorch_robotics_transformer.film_efficientnet.film_efficientnet_encoder_test
# If you want to test all test files in film_efficientnet directory, type below command on terminal.
# python -m unittest discover -s pytorch_robotics_transformer/film_efficientnet -p "*_test.py"

resize = 300
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# This transformation is for imagenet. Use another transformatioin for RT-1
img_trasnform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

class FilmEfficientnetTest(parameterized.TestCase, unittest.TestCase):
    # This test method will be implemented twice. One is implemented with include_film = True. Another is implemented with include_film = False.
    # To know further more, see a comment of https://github.com/abseil/abseil-py/blob/main/absl/testing/parameterized.py
    @parameterized.parameters([True, False])
    def test_film_efficientnet(self, include_film):
        fe =  EfficientNetB3(include_top=True, weights='imagenet', include_film=include_film)
        fe.eval()

        image = data.chelsea() # ndarray
        image_transformed = img_trasnform(image) # tensor

        # Show the image.
        # image_show = image_transformed.numpy().transpose((1, 2, 0))
        # image_show = np.clip(image_show, 0, 1)
        # plt.imshow(image_show)
        # plt.show()

        image_transformed = image_transformed.unsqueeze(0)
        context = torch.rand(1, 512)

        if include_film:
            eff_output = fe(image_transformed, context)
        else:
            eff_output = fe(image_transformed)
        
        predictor = ILSVRCPredictor(top=10)

        film_preds = predictor.predict_topk(eff_output)
        # print(film_preds)
        self.assertIn('tabby', film_preds)

if __name__ == '__main__':
    unittest.main()