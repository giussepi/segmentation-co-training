# -*- coding: utf-8 -*-

import unittest

import torch

from nns.utils import Normalizer


class Test_Normalizer(unittest.TestCase):

    def test_process_float_tensor(self):
        tensor = torch.rand(4, 3, 2, 2)

        ntensor = tensor.clone()

        shape = ntensor.size()
        ntensor = ntensor.view(shape[0], -1)
        ntensor -= ntensor.min(1, keepdim=True)[0]
        ntensor /= ntensor.max(1, keepdim=True)[0]
        ntensor = ntensor.view(shape)

        self.assertTrue(torch.equal(
            ntensor, Normalizer()(tensor)
        ))

    def test_process_float_tensor_in_place(self):
        tensor = torch.rand(4, 3, 2, 2)

        ntensor = tensor.clone()

        shape = ntensor.size()
        ntensor = ntensor.view(shape[0], -1)
        ntensor -= ntensor.min(1, keepdim=True)[0]
        ntensor /= ntensor.max(1, keepdim=True)[0]
        ntensor = ntensor.view(shape)

        Normalizer()(tensor, in_place=True)

        self.assertTrue(torch.equal(ntensor, tensor))

    def test_process_int_tensor(self):
        tensor = torch.randint(0, 10, (4, 3, 2, 2))

        ntensor = tensor.clone().float()

        shape = ntensor.size()
        ntensor = ntensor.view(shape[0], -1)
        ntensor -= ntensor.min(1, keepdim=True)[0]
        ntensor /= ntensor.max(1, keepdim=True)[0]
        ntensor = ntensor.view(shape)

        self.assertTrue(torch.equal(
            ntensor, Normalizer()(tensor)
        ))

    def test_process_int_tensor_in_place(self):
        tensor = torch.randint(0, 10, (4, 3, 2, 2))

        with self.assertRaises(AssertionError):
            Normalizer()(tensor, in_place=True)


if __name__ == '__main__':
    unittest.main()
