from typing import List, Optional

import torch
import torch.nn as nn


class BaseBlocks(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channels: List[int],
        kernels=None,
        strides=None,
        paddings=None,
        conv_type=None,
        activ_type=None,
        bn_type=None,
    ):
        if kernels is not None:
            assert len(out_channels) == len(kernels)
        if strides is not None:
            assert len(out_channels) == len(strides)
        if paddings is not None:
            assert len(out_channels) == len(paddings)

        super().__init__()
        in_channels = [in_channel, *out_channels[:-1]]

        blocks = []
        for block_i, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            is_last_block = block_i == len(out_channels) - 1
            conv_params = {}
            if kernels is not None:
                conv_params["kernel_size"] = kernels[block_i]
            if strides is not None:
                conv_params["stride"] = strides[block_i]
            if paddings is not None:
                conv_params["padding"] = paddings[block_i]

            if is_last_block:
                block = nn.Sequential(conv_type(in_ch, out_ch, **conv_params))
            else:
                if bn_type is None:
                    block = nn.Sequential(
                        conv_type(in_ch, out_ch, **conv_params),
                        activ_type(),
                    )
                else:
                    block = nn.Sequential(
                        conv_type(in_ch, out_ch, **conv_params),
                        bn_type(out_ch),
                        activ_type(),
                    )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Mlp(BaseBlocks):
    def __init__(
        self,
        in_channel,
        out_channels,
        norm_batch=False,
        active_type: Optional[nn.Module] = None,
    ):
        if active_type is None:
            active_type = nn.ReLU
        super().__init__(
            in_channel,
            out_channels,
            kernels=None,
            strides=None,
            paddings=None,
            conv_type=nn.Linear,
            activ_type=active_type,
            bn_type=nn.BatchNorm1d if norm_batch else None,
        )


class Conv2dNet(BaseBlocks):
    def __init__(
        self,
        in_channel,
        out_channels,
        kernels=None,
        strides=None,
        paddings=None,
        **kwargs,
    ):
        super().__init__(
            in_channel,
            out_channels,
            kernels,
            strides,
            paddings,
            conv_type=nn.Conv2d,
            activ_type=nn.ReLU,
            bn_type=nn.BatchNorm2d,
            **kwargs,
        )


class Conv2dTransposeNet(BaseBlocks):
    def __init__(
        self,
        in_channel,
        out_channels,
        kernels=None,
        strides=None,
        paddings=None,
    ):
        super().__init__(
            in_channel,
            out_channels,
            kernels,
            strides,
            paddings,
            conv_type=nn.ConvTranspose2d,
            activ_type=nn.LeakyReLU,
            bn_type=nn.BatchNorm2d,
        )


def weight_copy(module_a, module_b):
    for param_a, param_b in zip(module_a.parameters(), module_b.parameters()):
        param_a.data = param_b.data
    return module_a


if __name__ == "__main__":
    conv2dnet = Conv2dNet(3, [4, 5, 6], [3, 3, 3])
    print(conv2dnet)

    conv2dnet_ker = Conv2dNet(3, [4, 5, 6], [4, 4, 5], [2, 2, 2])
    print(conv2dnet_ker)
    out = conv2dnet_ker(torch.rand(4, 3, 128, 128))
    print(out.shape)

    mlp = Mlp(3, [4, 5, 6])
    print(mlp)
