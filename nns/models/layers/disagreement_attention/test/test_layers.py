# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/test/test_layers """

import unittest

import torch
from gtorch_utils.nns.utils import Normalizer
from gutils.exceptions.common import ExclusiveArguments

from nns.models.layers.disagreement_attention.constants import AttentionMergingType
from nns.models.layers.disagreement_attention.layers import AttentionMerging, AttentionMergingBlock


class Test_AttentionMerging(unittest.TestCase):
    def setUp(self):
        self.att1 = torch.rand(1, 3, 2, 2)
        self.att2 = torch.rand(1, 3, 2, 2)

    def test_sum(self):
        self.assertTrue(torch.equal(
            torch.sigmoid(self.att1 + self.att2),
            AttentionMerging(AttentionMergingType.SUM)(self.att1, self.att2)
        ))

    def test_max(self):
        self.assertTrue(torch.equal(
            self.att1.max(self.att2),
            AttentionMerging(AttentionMergingType.MAX)(self.att1, self.att2)
        ))

    def test_hadamard_only(self):
        self.assertTrue(torch.equal(
            self.att1 * self.att2,
            AttentionMerging(AttentionMergingType.HADAMARD)(self.att1, self.att2)
        ))

    def test_hadamard_with_sigmoid(self):
        self.assertTrue(torch.equal(
            torch.sigmoid(self.att1 * self.att2),
            AttentionMerging(AttentionMergingType.HADAMARD)(self.att1, self.att2, sigmoid=True)
        ))

    def test_hadamard_with_normalization(self):
        self.assertTrue(torch.equal(
            Normalizer()(self.att1 * self.att2),
            AttentionMerging(AttentionMergingType.HADAMARD)(self.att1, self.att2, normalize=True)
        ))

    def test_hadamard_exclusive_arguments(self):
        with self.assertRaises(ExclusiveArguments):
            AttentionMerging(AttentionMergingType.HADAMARD)(
                self.att1, self.att2, normalize=True, sigmoid=True)


class Test_AttentionMergingBlock(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(4, 3, 2, 2)
        self.gsa = torch.rand(4, 1, 2, 2)
        self.da = torch.rand(4, 1, 2, 2)

    def test_forward(self):
        amb = AttentionMergingBlock(6, 3, merging_type=AttentionMergingType.SUM)
        amb2 = AttentionMergingBlock(6, 3, merging_type=AttentionMergingType.MAX)
        x_att = amb(self.x, self.gsa, self.da)
        self.assertEqual(x_att.size(), self.x.size())
        self.assertFalse(torch.equal(self.x, x_att))
        self.assertFalse(torch.equal(x_att, amb2(self.x, self.gsa, self.da)))

    def test_forward_without_da(self):
        amb = AttentionMergingBlock(6, 3, merging_type=AttentionMergingType.SUM)
        amb2 = AttentionMergingBlock(6, 3, merging_type=AttentionMergingType.MAX)
        x_att = amb(self.x, self.gsa, None)
        self.assertEqual(x_att.size(), self.x.size())
        self.assertFalse(torch.equal(self.x, x_att))
        self.assertFalse(torch.equal(x_att, amb2(self.x, self.gsa, None)))

    def test_forward_with_disabled_da(self):
        amb = AttentionMergingBlock(6, 3, merging_type=AttentionMergingType.SUM)
        x_att_no_da = amb(self.x, self.gsa, self.da, disable_da=True)
        x_att2 = amb(self.x, self.gsa, self.da, disable_da=False)
        self.assertEqual(x_att_no_da.size(), self.x.size())
        self.assertFalse(torch.equal(self.x, x_att_no_da))
        self.assertFalse(torch.equal(x_att_no_da, x_att2))


if __name__ == '__main__':
    unittest.main()
