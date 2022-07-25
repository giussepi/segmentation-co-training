# -*- coding: utf-8 -*-
""" nns/models/test/test_x_attention_unet """

from copy import deepcopy
import unittest

import torch

from nns.models import XAttentionUNet


class Test_XAttentionUNet(unittest.TestCase):

    def setUp(self):
        self.input_ = torch.rand(2, 3, 256, 256, requires_grad=True)
        self.target = torch.randint(0, 2, (2, 3, 256, 256), dtype=torch.float32)

    def test_forward(self):
        model = XAttentionUNet(3, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial_weights = deepcopy(model.state_dict())

        pred = model(self.input_)
        loss = torch.nn.functional.cross_entropy(pred, self.target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for ini, final in zip(initial_weights.items(), model.state_dict().items()):
            self.assertEqual(ini[0], final[0])
            self.assertFalse(torch.equal(ini[1], final[1]))


if __name__ == '__main__':
    unittest.main()
