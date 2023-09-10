import torch.nn as nn
import random
from .SuperNetBlocks import ConvBlock, VariableConvBlock


class SuperNet(nn.Module):
    """
    SuperNet Class for Neural Architecture Search.

    :param config: Configuration dictionary containing layer parameters
    :type config: dict
    """

    def __init__(self, config):
        super(SuperNet, self).__init__()

        self.init_conv = ConvBlock(**config["init_conv"])
        self.variable_block1 = VariableConvBlock(**config["variable_block1"])
        self.downsample_conv = ConvBlock(**config["downsample_conv"])
        self.variable_block2 = VariableConvBlock(**config["variable_block2"])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(**config["fc"])

        self.layer_config = (1, 1)

    def sampler(self, config=None):
        """
        Samples a specific sub-network architecture or a random one.

        :param config: Tuple indicating the number of layers for each variable block, defaults to None
        :type config: tuple, optional
        """
        if config and not config["random"]:
            self.layer_config = config["fixed_config"]
        else:
            self.layer_config = (random.randint(1, 3), random.randint(1, 3))


    def forward(self, x):
        """
        Forward pass of the SuperNet.

        :param x: Input tensor
        :type x: torch.Tensor
        :rtype: torch.Tensor
        :return: Model's output tensor
        """
        x = self.init_conv(x)
        x = self.variable_block1(x, self.layer_config[0])
        x = self.downsample_conv(x)
        x = self.variable_block2(x, self.layer_config[1])
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
