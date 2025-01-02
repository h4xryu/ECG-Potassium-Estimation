import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Callable, Union
import warnings
import torch.nn.functional as F

def same_padding_conv1d(input_tensor, conv_layer):
    # 커널 크기와 스트라이드 가져오기
    kernel_size = conv_layer.kernel_size[0]
    stride = conv_layer.stride[0]

    # 입력 길이
    input_length = input_tensor.shape[-1]

    # 패딩 계산
    padding_needed = max((input_length - 1) * stride + kernel_size - input_length, 0)
    pad_left = padding_needed // 2
    pad_right = padding_needed - pad_left

    # 패딩 추가
    padded_input = F.pad(input_tensor, (pad_left, pad_right), mode='constant', value=0)

    # Conv1D 적용
    return conv_layer(padded_input)

class _SeparableConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(_SeparableConv, self).__init__()

        self.dwconv = None
        self.dwconv_normalization = None
        self.dwconv_activation = None

        self.pwconv = None
        self.pwconv_normalization = None
        self.pwconv_activation = None

    def forward(self, x):
        assert self.dwconv is not None and self.pwconv is not None, (
            "Depthwise Convolution and/or Pointwise Convolution is/are not implemented"
            " yet."
        )

        x = self.dwconv(x)

        if self.dwconv_normalization is not None:
            x = self.dwconv_normalization(x)

        if self.dwconv_activation is not None:
            x = self.dwconv_activation(x)

        x = self.pwconv(x)

        if self.pwconv_normalization is not None:
            x = self.pwconv_normalization(x)

        if self.dwconv_activation is not None:
            x = self.pwconv_activation(x)

        return x

class SeparableConv1d(_SeparableConv):
    r"""Applies a 1D depthwise separable convolution over an input signal composed of several input
    planes as described in the paper
    `Xception: Deep Learning with Depthwise Separable Convolutions <https://arxiv.org/abs/1610.02357>`__ .

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        normalization_dw (str, optional): depthwise convolution normalization. Default: 'bn'
        normalization_pw (str): pointwise convolution normalization. Default: 'bn'
        activation_dw (Callable[..., torch.nn.Module], optional): depthwise convolution activation. Default: ``torch.nn.ReLU``
        activation_pw (Callable[..., torch.nn.Module], optional): pointwise convolution activation. Default: ``torch.nn.ReLU``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        padding_mode: str = "zeros",
        dilation: _size_1_t = 1,
        depth_multiplier: int = 1,
        normalization_dw: str = "bn",
        normalization_pw: str = "bn",
        activation_dw: Callable[..., nn.Module] = nn.ReLU,
        activation_pw: Callable[..., nn.Module] = nn.ReLU,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super(SeparableConv1d, self).__init__()

        expansion_channels = max(in_channels * int(depth_multiplier), in_channels)

        if in_channels * depth_multiplier != expansion_channels:
            raise ValueError("depth_multiplier must be integer>=1")

        self.dwconv = nn.Conv1d(
            in_channels,
            expansion_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.BatchNorm1d(expansion_channels)
            if normalization_dw == "bn"
            else nn.InstanceNorm1d(expansion_channels)
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm1d`` or 'in' for ``nn.InstanceNorm1d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.Conv1d(
            expansion_channels,
            out_channels,
            1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.pwconv_normalization = (
            nn.BatchNorm1d(out_channels)
            if normalization_pw == "bn"
            else nn.InstanceNorm1d(out_channels)
            if normalization_pw == "in"
            else None
        )

        if self.pwconv_normalization is None:
            warnings.warn(
                "normalization_pw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm1d`` or 'in' for ``nn.InstanceNorm1d``."
            )

        self.pwconv_activation = activation_pw()

class DSCRNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSCRNN, self).__init__()

        self.SepConv1 = SeparableConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=128, padding=0)
        self.SepConv2 = SeparableConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=64, padding=0)



    def forward(self, x):
        x = same_padding_conv1d(x, self.SepConv1)
        x = same_padding_conv1d(x, self.SepConv2)




