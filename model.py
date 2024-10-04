import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=2):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels // reduction_rate, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels // reduction_rate, out_channels=channels, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.se(x) * x
    

class BaseLineBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, groups=num_channels)
        self.norm1 = nn.GroupNorm(16, num_channels)
        self.gelu = nn.GELU()
        self.CA = ChannelAttention(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 1, 1, 0)

        self.norm2 = nn.GroupNorm(16, num_channels)
        self.conv4 = nn.Conv2d(num_channels, num_channels*2, 1, 1, 0)
        self.conv5 = nn.Conv2d(num_channels*2, num_channels, 1, 1, 0)

        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)

    def forward(self, residual):
        x = self.norm1(residual)
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
        self.num_encoder_blocks = [1, 1, 1, 28]
        self.num_decoder_blocks = [1, 1, 1, 1]
        self.num_bottleneck_block = 1
        self.base_channels = 16
        
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
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(
                    *[BaseLineBlock(channels) for _ in range(num)]
                )
            )

    def forward(self, inp):
        x = self.in_conv(inp)
        skip_connections = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.downs[i](x)
            
        x = self.bottlenecks(x)
        for i in range(len(self.decoders)):
            x = self.ups[i](x)
            x = x + skip_connections[len(self.decoders)-i-1]
            x = self.decoders[i](x)
        x = self.out_conv(x)
        return x + inp
    
if __name__ == '__main__':
    from torchinfo import summary

    model = BaseLineUnet()
    summary(model, (16, 3, 256, 256))