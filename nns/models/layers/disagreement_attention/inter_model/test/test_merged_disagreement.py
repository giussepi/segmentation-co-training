# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/inter_model/test/test_merged_disagreement """

import unittest

import torch

from nns.models.layers.disagreement_attention.inter_model import MergedDisagreementAttentionBlock


class Test_MergedDisagreementAttentionBlock(unittest.TestCase):
    def setUp(self):
        self.activations1 = torch.randn(1, 4, 2, 2)
        self.activations2 = torch.randn(1, 4, 2, 2)

    def test_forward(self):
        act1_with_attention, att = MergedDisagreementAttentionBlock(4, 4)(self.activations1, self.activations2)
        self.assertFalse(torch.equal(act1_with_attention, self.activations1))
        self.assertFalse(torch.equal(act1_with_attention, self.activations2))
        self.assertTrue(self.activations1.shape == self.activations2.shape == act1_with_attention.shape)


if __name__ == '__main__':
    unittest.main()
