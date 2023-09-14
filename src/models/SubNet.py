import torch.nn as nn


class SubNet(nn.Module):
    """
    A SubNet class derived from the given SuperNet. This class helps to instantiate a subset of the SuperNet based on the provided layers.

    :param supernet: An instance of the SuperNet to derive layers from
    :type supernet: nn.Module
    :param layers_first: Number of layers to consider from the first variable block of the SuperNet
    :type layers_first: int
    :param layers_second: Number of layers to consider from the second variable block of the SuperNet
    :type layers_second: int
    """
    def __init__(self, supernet, layers_first, layers_second):
        super(SubNet, self).__init__()
        self.init_conv = supernet.init_conv

        # Copy only the required layers
        self.variable_block1 = nn.ModuleList(
            supernet.variable_block1[:layers_first]
        )
        self.downsample_conv = supernet.downsample_conv
        self.variable_block2 = nn.ModuleList(
            supernet.variable_block2[:layers_second]
        )

        self.global_avg_pool = supernet.global_avg_pool
        self.fc = supernet.fc

    def forward(self, x):
        """
        Forward pass of the SubNet.

        :param x: Input tensor
        :type x: torch.Tensor
        :rtype: torch.Tensor
        :return: Model's output tensor
        """
        x = self.init_conv(x)

        for layer in self.variable_block1:
            x = layer(x)

        x = self.downsample_conv(x)

        for layer in self.variable_block2:
            x = layer(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
