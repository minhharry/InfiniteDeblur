import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super().__init__()
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels // reduction_rate,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // reduction_rate,
                      out_channels=channels,
                      kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # perform squeeze with independent Pooling
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        # perform excitation with the same excitation sub-net
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out)
        return attention * x
    

class BaseLineBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(num_channels)
        self.gelu = nn.GELU()
        self.CA = ChannelAttention(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 1, 1, 0)

        self.norm2 = nn.InstanceNorm2d(num_channels)
        self.conv4 = nn.Conv2d(num_channels, num_channels*2, 1, 1, 0)
        self.conv5 = nn.Conv2d(num_channels*2, num_channels, 1, 1, 0)

        self.beta = nn.Parameter(torch.randn(1, num_channels, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(1, num_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.CA(x)
        x = self.conv3(x)

        y = x * self.beta + residual

        x = self.norm2(x)
        x = self.conv4(x)
        x = self.gelu(x)
        x = self.conv5(x)

        return x * self.gamma + y
    
class BaseLineUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_encoder_blocks = [1, 2, 4, 8]
        self.num_decoder_blocks = [1, 1, 1, 1]
        self.num_bottleneck_block = 2
        self.base_channels = 32
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        channels = self.base_channels

        self.in_conv = nn.Conv2d(3, channels, 3, 1, 1)
        self.out_conv = nn.Conv2d(channels, 3, 3, 1, 1)

        for num in self.num_encoder_blocks:
            self.encoders.append(
                nn.Sequential(
                    *[BaseLineBlock(channels) for _ in range(num)]
                )
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, 2))
            channels *= 2

        self.bottlenecks = nn.Sequential(
            *[BaseLineBlock(channels) for _ in range(self.num_bottleneck_block)]
        )

        for num in self.num_decoder_blocks:
            
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear"),
                    nn.Conv2d(channels, channels // 2, 3, 1, 1),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(
                    *[BaseLineBlock(channels) for _ in range(num)]
                )
            )

    def forward(self, x):
        x = self.in_conv(x)
        skip_connections = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.downs[i](x)
            
        x = self.bottlenecks(x)
        for i in range(len(self.decoders)):
            x = self.ups[i](x)
            x = x + skip_connections.pop()
            x = self.decoders[i](x)
        x = self.out_conv(x)
        return x
    
if __name__ == '__main__':
    from torchinfo import summary

    model = BaseLineUnet()
    summary(model, (16, 3, 256, 256))