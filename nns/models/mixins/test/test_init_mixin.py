# -*- coding: utf-8 -*-
""" nns/models/mixins/test/test_init_mixin """

import unittest
from unittest.mock import patch, call

from torch import nn
from gtorch_utils.nns.models.segmentation.unet3_plus.constants import UNet3InitMethod


class Test_InitMixin(unittest.TestCase):

    @patch('nns.models.mixins.init_mixin.init_weights')
    @patch('nns.models.mixins.init_mixin.UNet3InitMethod')
    def test_defaults(self, Mockunet3initmethod, Mock_init_weights):
        from nns.models.mixins import InitMixin

        class MyModel(nn.Module, InitMixin):

            def __init__(self):
                super().__init__()
                self.w1 = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1)
                )
                self.w2 = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1)
                )
                self.attention_2to1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
                )

                self.initialize_weights()

        model = MyModel()
        Mockunet3initmethod.validate.assert_called_once_with(UNet3InitMethod.KAIMING)

        with self.assertRaises(AssertionError):
            Mock_init_weights.assert_not_called()

        calls = []

        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                calls.append(call(module, init_type=UNet3InitMethod.KAIMING))

        self.assertEqual(len(calls), 6)
        Mock_init_weights.assert_has_calls(calls, any_order=True)

    @patch('nns.models.mixins.init_mixin.init_weights')
    @patch('nns.models.mixins.init_mixin.UNet3InitMethod')
    def test_non_defaults(self, Mockunet3initmethod, Mock_init_weights):
        from nns.models.mixins import InitMixin

        class MyModel(nn.Module, InitMixin):

            def __init__(self, init_type, layers_cls):
                super().__init__()
                self.w1 = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1)
                )
                self.w2 = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1)
                )
                self.attention_2to1 = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
                )

                self.initialize_weights(init_type, layers_cls=layers_cls)

        layers_cls = (nn.Conv2d,)
        model = MyModel(UNet3InitMethod.NORMAL, layers_cls)
        Mockunet3initmethod.validate.assert_called_once_with(UNet3InitMethod.NORMAL)

        with self.assertRaises(AssertionError):
            Mock_init_weights.assert_not_called()

        calls = []

        for module in model.modules():
            if isinstance(module, layers_cls):
                calls.append(call(module, init_type=UNet3InitMethod.NORMAL))

        self.assertEqual(len(calls), 3)
        Mock_init_weights.assert_has_calls(calls, any_order=True)


if __name__ == '__main__':
    unittest.main()
