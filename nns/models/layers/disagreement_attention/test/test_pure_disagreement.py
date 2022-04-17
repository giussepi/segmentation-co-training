# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/test/test_pure_disagreement """

import unittest

import torch

from nns.models.layers.disagreement_attention import PureDisagreementAttentionBlock


class Test_PureDisagreementAttentionBlock(unittest.TestCase):

    def setUp(self):
        self.activations1 = torch.randn(1, 2, 3, 3)
        self.activations2 = torch.randn(1, 2, 3, 3)

    def test_forward(self):
        activations1_with_attention, att = PureDisagreementAttentionBlock(2, 2)(
            self.activations1, self.activations2)
        self.assertFalse(torch.equal(self.activations1, activations1_with_attention))
        self.assertFalse(torch.equal(self.activations2, activations1_with_attention))
        self.assertTrue(self.activations1.shape == self.activations2.shape == activations1_with_attention.shape)


if __name__ == '__main__':
    unittest.main()
