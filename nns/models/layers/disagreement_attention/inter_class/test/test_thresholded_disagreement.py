# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/inter_class/test/test_thresholded_disagreement """

import unittest

import torch

from nns.models.layers.disagreement_attention.inter_class import ThresholdedDisagreementAttentionBlock


class Test_ThresholdedDisagreementAttentionBlock(unittest.TestCase):
    def setUp(self):
        self.act1 = torch.randn(1, 3, 2, 2)
        self.act2 = torch.randn(1, 3, 2, 2)

    def test_forward_without_beta(self):
        act1_with_attention, attention = ThresholdedDisagreementAttentionBlock(3, 3)(self.act1, self.act2)
        self.assertFalse(torch.equal(act1_with_attention, self.act1))
        self.assertFalse(torch.equal(act1_with_attention, self.act2))
        self.assertTrue(self.act1.shape == self.act2.shape == act1_with_attention.shape)

    def test_forward_with_beta(self):
        act1_with_attention, attention = ThresholdedDisagreementAttentionBlock(3, 3, beta=.3)(self.act1, self.act2)
        self.assertFalse(torch.equal(act1_with_attention, self.act1))
        self.assertFalse(torch.equal(act1_with_attention, self.act2))
        self.assertTrue(self.act1.shape == self.act2.shape == act1_with_attention.shape)


if __name__ == '__main__':
    unittest.main()
