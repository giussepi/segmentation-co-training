# -*- coding: utf-8 -*-
""" nns/models/layers/disagreement_attention/layers """

from typing import Union

import torch

from gutils.exceptions.common import ExclusiveArguments
from nns.models.layers.disagreement_attention.constants import AttentionMergingType
from nns.utils import Normalizer


__all__ = ['ConvBlock', 'DAConvBlock', 'AttentionMerging', 'AttentionMergingBlock']


class ConvBlock(torch.nn.Module):
    """ Convolutional block to be used after a disagreement attention block """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert isinstance(in_channels, int), type(in_channels)
        assert isinstance(out_channels, int), type(out_channels)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), type(x)

        x = self.conv(x)

        return x


class DAConvBlock(torch.nn.Module):
    """
    Disagreement attention convolutional block

    Usage:
         class UNet_3Plus_DA(UNet_3Plus):
            def __init__(self, in_channels=3, out_channels=10):
                super().__init__(in_channels, out_channels)
                # placing disagreement attention between mlpconv1 and mlpconv2
                self.daconvblock1 = DAConvBlock(ThresholdedDisagreementAttentionBlock(96, 96), 192, 96)
                # placing disagreement attention between mlpconv2 and mlpconv3
                self.daconvblock2 = DAConvBlock(ThresholdedDisagreementAttentionBlock(192, 192), 384, 192)
    """

    def __init__(self, dablock_obj: torch.nn.Module, conv_in_channels: int, conv_out_channels: int, /, *,
                 only_attention: bool = False):
        """
        Kwargs:
            dablock <torch.nn.Module>: Disagreement attention block instance.
                                       e.g ThresholdedDisagreementAttentionBlock(96, 96), 192, 96)
            conv_in_channels    <int>: ConvBlock in channels
            conv_out_channels   <int>: ConvBlock out channels
            only_attention     <bool>: If true returns only the attention; otherwise, returns the
                                       activation maps with attention. Default False
        """
        super().__init__()
        assert isinstance(dablock_obj, torch.nn.Module), \
            'The provided dablock_obj is not an instance of torch.nn.Module'
        assert isinstance(conv_in_channels, int), type(conv_in_channels)
        assert isinstance(conv_out_channels, int), type(conv_out_channels)
        assert isinstance(only_attention, bool), type(only_attention)

        self.dattentionblock = dablock_obj
        self.identity = torch.nn.Identity()
        self.only_attention = only_attention

        if not self.only_attention:
            self.convblock = ConvBlock(conv_in_channels, conv_out_channels)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor, /, *, disable_attention: bool = False):
        """
        Kwargs:
            x               <torch.Tensor>: activation/feature maps
            skip_connection <torch.Tensor>: skip connection containing activation/feature maps
            disable_attention       <bool>: When set to True, identity(x) will be used instead of
                                        dattentionblock(x, skip_connection). Default False

        Returns:
            Union[torch.Tensor, None]
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(skip_connection, torch.Tensor), type(skip_connection)
        assert isinstance(disable_attention, bool), type(disable_attention)

        if disable_attention:
            da, att = self.identity(x), None
        else:
            da, att = self.dattentionblock(x, skip_connection)

        if self.only_attention:
            return att

        x = torch.cat((da, x), dim=1)
        x = self.convblock(x)

        return x


class AttentionMerging:
    """
    Holds methods to merge two attention tensors

    Usage:
        AttentionMerging(AttentionMergingType.SUM)(att1, att2)
    """

    def __init__(self, type_: str = AttentionMergingType.SUM):
        """
        Kwargs:
            type_ <str>: Valid attention merging strategy
                         (see nns.models.layers.disagreement_attention.constants.AttentionMergingType).
                         Default AttentionMergingType.SUM
        """
        AttentionMergingType.validate(type_)
        self.type_ = type_

    def __call__(self, att1: torch.Tensor, att2: torch.Tensor, **kwargs):
        """
        Kwargs:
            att1 <torch.Tensor>: attention 1
            att2 <torch.Tensor>: attention 2
        """
        return getattr(self, self.type_)(att1, att2, **kwargs)

    @staticmethod
    def sum(att1: torch.Tensor, att2: torch.Tensor):
        """
        Sums the tensor and applied sigmoid at the end

        Kwargs:
            att1 <torch.Tensor>: attention 1
            att2 <torch.Tensor>: attention 2
        """
        assert isinstance(att1, torch.Tensor), type(att1)
        assert isinstance(att2, torch.Tensor), type(att2)

        summation = att1 + att2
        summation = torch.sigmoid(summation)

        return summation

    @staticmethod
    def max(att1: torch.Tensor, att2: torch.Tensor):
        """
        Selects the highest values

        Kwargs:
            att1 <torch.Tensor>: attention 1
            att2 <torch.Tensor>: attention 2
        """
        assert isinstance(att1, torch.Tensor), type(att1)
        assert isinstance(att2, torch.Tensor), type(att2)

        max_values = att1.max(att2)

        return max_values

    @staticmethod
    def hadamard(att1: torch.Tensor, att2: torch.Tensor, normalize: bool = False, sigmoid: bool = False):
        """
        Kwargs:
            att1 <torch.Tensor>: attention 1
            att2 <torch.Tensor>: attention 2
            normalize    <bool>: if True normalization will be applied to the final product.
                                 This argument and sigmoid are exclusive.
                                 Default False
            sigmoid      <bool>: if True sigmoid will be applied to the final product
                                 This argument and normalize are exclusive.
                                 Default False
        """
        assert isinstance(att1, torch.Tensor), type(att1)
        assert isinstance(att2, torch.Tensor), type(att2)
        assert isinstance(normalize, bool), type(normalize)
        assert isinstance(sigmoid, bool), type(sigmoid)

        if normalize and sigmoid:
            raise ExclusiveArguments(['normalize', 'sigmoid'])

        hadamard_prod = att1 * att2

        # with normalizer the results are a bit similar to using only hadarmard but with min
        # and max values of 0,1
        if normalize:
            return Normalizer()(hadamard_prod)

        # with sigmoid the values align almost randomly so not the best option
        if sigmoid:
            return torch.sigmoid(hadamard_prod)

        # using only hadamard the values align to the smallest
        return hadamard_prod


class AttentionMergingBlock(torch.nn.Module):
    """
    Merges the provided attentions first, then calculate the input with attention and finally computes
    attention feature maps with the desired numer of channels

    Usage:
        attmerge = AttentionMergingBlock(2x, x)
    """

    def __init__(self,
                 conv_in_channels: int, conv_out_channels: int, /, *,
                 merging_type: str = AttentionMergingType.SUM):
        """
        Kwargs:
            conv_in_channels  <int>: ConvBlock in channels
            conv_out_channels <int>: ConvBlock out channels
            merging_type      <str>: Valid attention merging strategy
                                     (see nns.models.layers.disagreement_attention.constants.AttentionMergingType).
                                     Default AttentionMergingType.SUM
        """
        super().__init__()
        assert isinstance(conv_in_channels, int), type(conv_in_channels)
        assert isinstance(conv_out_channels, int), type(conv_out_channels)

        self.merging_strategy = AttentionMerging(merging_type)
        self.convblock = ConvBlock(conv_in_channels, conv_out_channels)

    def forward(
            self, x: torch.Tensor, gsa: torch.Tensor, da: Union[torch.Tensor, None], /, *,
            disable_da: bool = False
    ):
        """
        Kwargs:
            x   <torch.Tensor>: activation/feature maps where the attentions will be added
            gsa <torch.Tensor>: gating signal attention
            da  <torch.Tensor>: disagreement attention
            disable_da  <bool>: If True or da is None the merged_attentions is set to gsa; otherwise,
                                merged_attentions is computed as the combination of gsa and da.
                                Default False

        Returns:
            x_with_merged_attentions <torch.Tensor>
        """
        assert isinstance(x, torch.Tensor), type(x)
        assert isinstance(gsa, torch.Tensor), type(gsa)

        if da is not None:
            assert isinstance(da, torch.Tensor), type(da)
        assert isinstance(disable_da, bool), type(disable_da)

        # TODO: think how it could work when disable_da = True (this is necessary)
        #       to keep the models separable
        # merging atts before applying them ###################################
        if disable_da or da is None:
            # FIXME: I don't think this workaround will deliver similar results after
            #        separating the models... I need think and test a lot!
            #        It could work only when using MAX merging stragegy because
            #        with enough training the DA could become just a few pixels (negligible);
            #        thus, gsa.max(da) \simeq gsa
            merged_attentions = gsa
        else:
            merged_attentions = self.merging_strategy(gsa, da)

        x_with_merged_attentions = x * merged_attentions
        x = torch.cat((x, x_with_merged_attentions), dim=1)
        x = self.convblock(x)

        # TODO: we can merge like this
        # in this case con_in_channels would be + 2 e.g. AttentionMergingBlock(x+2, x) ?????

        # TODO: or like this. in_channels would be + 2 e.g. AttentionMergingBlock(x+1, x) ?????
        # x = torch.cat((x, AttentionMerging(AttentionMergingType.SUM)(gsa, da)). dim=1)
        # TODO: another way would be using the DA of X and the upsampled gating signal,
        #       merge them, concatenate. We would use AttentionMergingBlock(2x, x)
        # TODO: one last way following the previous one would be not merging the DA of x with
        #       the gating signal to we would use AttentionMergingBlock(3x, x)

        return x
