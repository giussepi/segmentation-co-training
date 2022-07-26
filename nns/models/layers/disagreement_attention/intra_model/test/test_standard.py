# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/intra_model/test/test_standard """

import unittest

import torch

from nns.models.layers.disagreement_attention.intra_model import AttentionBlock


class Test_AttentionBlock(unittest.TestCase):

    def setUp(self):
        self.input_ = torch.rand(2, 4, 4, 4)
        self.gated_signal = torch.rand(2, 8, 2, 2)

    def test_forward(self):
        g = AttentionBlock(4, 8)
        input_with_gs_att, att = g(self.input_, self.gated_signal)

        self.assertEqual(input_with_gs_att.shape, self.input_.shape)
        self.assertFalse(torch.equal(input_with_gs_att, self.input_))
        self.assertEqual(att.shape, (2, 1, 4, 4))

    def test_forward_with_custom_n_channels(self):
        g = AttentionBlock(4, 8, n_channels=10)
        input_with_gs_att, att = g(self.input_, self.gated_signal)

        self.assertEqual(input_with_gs_att.shape, self.input_.shape)
        self.assertFalse(torch.equal(input_with_gs_att, self.input_))
        self.assertEqual(att.shape, (2, 1, 4, 4))


if __name__ == '__main__':
    unittest.main()
