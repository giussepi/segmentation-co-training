# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/inter_model/test/test_embedded_disagreement """

import unittest
from functools import reduce

import torch
from torch.nn.functional import interpolate

from nns.models.layers.disagreement_attention.inter_model import EmbeddedDisagreementAttentionBlock


def downsample(x: torch.Tensor):
    assert isinstance(x, torch.Tensor)
    return interpolate(x, scale_factor=.5, mode='bilinear', align_corners=False)


class Test_EmbeddedDisagreementAttentionBlock(unittest.TestCase):

    def test_forward(self):
        act1 = torch.randn(1, 2, 3, 3)
        act2 = torch.randn(1, 2, 3, 3)

        act1_with_attention, attention = EmbeddedDisagreementAttentionBlock(
            act1.size(1), act2.size(1))(act1, act2)

        self.assertIsInstance(act1_with_attention, torch.Tensor)
        self.assertEqual(act1_with_attention.nelement(), reduce((lambda x, y: x*y), act1.shape))
        self.assertEqual(act1_with_attention.size(), act1.size())
        self.assertIsInstance(attention, torch.Tensor)
        self.assertEqual(attention.nelement(), reduce((lambda x, y: x*y), act1.shape[2:]))
        self.assertEqual(attention.shape, (act1.size(0), 1, *act1.shape[2:]))

    def test_forward_act2_bigger_and_downsampled(self):
        act1 = torch.randn(1, 2, 3, 3)
        act2 = torch.randn(1, 2, 6, 6)

        act1_with_attention, attention = EmbeddedDisagreementAttentionBlock(
            act1.size(1), act2.size(1), resample=downsample)(act1, act2)

        self.assertIsInstance(act1_with_attention, torch.Tensor)
        self.assertEqual(act1_with_attention.nelement(), reduce((lambda x, y: x*y), act1.shape))
        self.assertEqual(act1_with_attention.size(), act1.size())
        self.assertIsInstance(attention, torch.Tensor)
        self.assertEqual(attention.nelement(), reduce((lambda x, y: x*y), act1.shape[2:]))
        self.assertEqual(attention.shape, (act1.size(0), 1, *act1.shape[2:]))

    def test_forward_act2_bigger_and_not_downsampled(self):
        act1 = torch.randn(1, 2, 3, 3)
        act2 = torch.randn(1, 2, 6, 6)

        with self.assertRaises(RuntimeError):
            act1_with_attention, attention = EmbeddedDisagreementAttentionBlock(
                act1.size(1), act2.size(1))(act1, act2)


if __name__ == '__main__':
    unittest.main()
