# -*- coding: utf-8 -*-
""" nns/segmentation/utils/postprocessing/test/test_expand_predictions """

import unittest

import torch

from nns.segmentation.utils.postprocessing.expand_predictions import ExpandPrediction


class Test_ExpandPrediction(unittest.TestCase):

    def setUp(self):
        self.predicted_mask = torch.Tensor([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
        ])

        self.sub_prediction = torch.Tensor([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        self.expected_mask_with_diagonal_pixels = torch.Tensor([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        ])

        self.expected_mask_no_diagonal_pixels = torch.Tensor([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    def test_small_grow(self):
        expected_grown_mask = torch.Tensor([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        ])
        self.assertTrue(torch.equal(
            expected_grown_mask,
            ExpandPrediction().small_grow(self.sub_prediction, self.predicted_mask)
        ))

    def test_small_grow_with_diagonal_pixels(self):
        expected_grown_mask = torch.Tensor([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])
        self.assertTrue(torch.equal(
            expected_grown_mask,
            ExpandPrediction(diagonal_pixels=True).small_grow(self.sub_prediction, self.predicted_mask)
        ))

    def test_process(self):
        self.assertFalse(torch.equal(
            self.expected_mask_with_diagonal_pixels,
            ExpandPrediction()(self.sub_prediction, self.predicted_mask)
        ))
        self.assertFalse(torch.equal(
            self.predicted_mask,
            ExpandPrediction()(self.sub_prediction, self.predicted_mask)
        ))

        self.assertTrue(torch.equal(
            self.expected_mask_no_diagonal_pixels,
            ExpandPrediction()(self.sub_prediction, self.predicted_mask)
        ))

    def test_process_with_diagonal_pixels(self):
        self.assertFalse(torch.equal(
            self.expected_mask_no_diagonal_pixels,
            ExpandPrediction(diagonal_pixels=True)(self.sub_prediction, self.predicted_mask)
        ))
        self.assertFalse(torch.equal(
            self.predicted_mask,
            ExpandPrediction(diagonal_pixels=True)(self.sub_prediction, self.predicted_mask)
        ))
        self.assertTrue(torch.equal(
            self.expected_mask_with_diagonal_pixels,
            ExpandPrediction(diagonal_pixels=True)(self.sub_prediction, self.predicted_mask)
        ))


if __name__ == '__main__':
    unittest.main()
