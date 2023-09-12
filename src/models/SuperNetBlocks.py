import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional Block consisting of Conv2D, BatchNorm, and ReLU.

    :param in_channels: Number of input channels
    :type in_channels: int
    :param out_channels: Number of output channels
    :type out_channels: int
    :param kernel_size: Size of the convolutional kernel, defaults to 3
    :type kernel_size: int, optional
    :param padding: Padding for the convolutional layer, defaults to 1
    :type padding: int, optional
    :param stride: Stride for the convolutional layer, defaults to 1
    :type stride: int, optional
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass for the ConvBlock.

        :param x: Input tensor
        :type x: torch.Tensor
        :rtype: torch.Tensor
        :return: Output tensor after passing through the block
        """
        return self.block(x)