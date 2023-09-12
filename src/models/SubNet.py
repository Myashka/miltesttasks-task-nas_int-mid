import torch.nn as nn


class SubNet(nn.Module):
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
