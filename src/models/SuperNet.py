import torch.nn as nn
import random
from .SuperNetBlocks import ConvBlock, SubNet


class SuperNet(nn.Module):
    """
    SuperNet Class for Neural Architecture Search.

    :param config: Configuration dictionary containing layer parameters
    :type config: dict
    """

    def __init__(self, config):
        super(SuperNet, self).__init__()

        # Initial Conv
        self.init_conv = ConvBlock(**config["init_conv"])
        
        # Variable Blocks as lists
        self.variable_block1 = nn.ModuleList([
            ConvBlock(**config["variable_block1"]),
            ConvBlock(**config["variable_block1"]),
            ConvBlock(**config["variable_block1"])
        ])
        
        self.downsample_conv = ConvBlock(**config["downsample_conv"])
        
        self.variable_block2 = nn.ModuleList([
            ConvBlock(**config["variable_block2"]),
            ConvBlock(**config["variable_block2"]),
            ConvBlock(**config["variable_block2"])
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(**config["fc"])

        self.layer_config = (1, 1)

    def sampler(self, config=None):
        """
        Samples a specific sub-network architecture or a random one.

        :param config: Dictionary indicating the number of layers for each variable block or randomness, defaults to None
        :type config: dict, optional
        """
        if config and not config.get("random"):
            layers_first, layers_second = config["fixed_config"]
        else:
            layers_first = random.randint(1, 3)
            layers_second = random.randint(1, 3)
        
        return SubNet(self, layers_first, layers_second)


    def forward(self, x):
        """
        Forward pass of the SuperNet.

        :param x: Input tensor
        :type x: torch.Tensor
        :rtype: torch.Tensor
        :return: Model's output tensor
        """
        x = self.init_conv(x)
        x = self.variable_block1(x, int(self.layer_config[0]))
        x = self.downsample_conv(x)
        x = self.variable_block2(x, int(self.layer_config[1]))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
