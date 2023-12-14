import torch
import torch.nn as nn

class ResidualDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.identity_conv = nn.Conv2d(in_channels, out_channels, 1)  # Identity mapping
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.identity_conv(x)
        x = self.conv(x)
        x += identity  # Adding the residual connection
        x = self.relu(x)
        x = self.pool(x)
        return x

class AttentionUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class FinalUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalUpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = ResidualDownsampleBlock(3, 64)
        self.down2 = ResidualDownsampleBlock(64, 128)
        self.down3 = ResidualDownsampleBlock(128, 256)
        self.down4 = ResidualDownsampleBlock(256, 512)

        self.up1 = AttentionUpsampleBlock(512, 256)
        self.up2 = AttentionUpsampleBlock(256, 128)
        self.up3 = AttentionUpsampleBlock(128, 64)
        self.up4 = FinalUpsampleBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x)
        x = self.out_conv(x)
        return x

class DilatedConvBlock(nn.Module):
    # Dilated convolution block
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DeepResidualDownsampleBlock(nn.Module):
    # Increase the out_channels in each block
    def __init__(self, in_channels, out_channels, increase_factor=2):
        super(DeepResidualDownsampleBlock, self).__init__()
        increased_channels = out_channels * increase_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, increased_channels, 3, padding=1),
            nn.BatchNorm2d(increased_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(increased_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.identity_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.identity_conv(x)
        x = self.conv(x)
        x += identity
        x = self.relu(x)
        x = self.pool(x)
        return x


class DeepUNet(nn.Module):
    def __init__(self):
        super(DeepUNet, self).__init__()
        self.down1 = DeepResidualDownsampleBlock(3, 64)
        self.down2 = DeepResidualDownsampleBlock(64, 128)
        self.down3 = DeepResidualDownsampleBlock(128, 256)
        self.down4 = DeepResidualDownsampleBlock(256, 512)

        self.dilated = DilatedConvBlock(512, 512, dilation_rate=2)  # Adding dilated convolution block

        self.up1 = AttentionUpsampleBlock(512, 256)
        self.up2 = AttentionUpsampleBlock(256, 128)
        self.up3 = AttentionUpsampleBlock(128, 64)
        self.up4 = FinalUpsampleBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x4 = self.dilated(x4)  # Apply dilated convolution

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x)
        x = self.out_conv(x)
        return x

class AdvancedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdvancedUpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels * 4, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * out_channels * 4, out_channels * 4, 3, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Additional convolution
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ExtraLayersBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExtraLayersBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Additional layers
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class VeryDeepResidualDownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VeryDeepResidualDownsampleBlock, self).__init__()
        # Increase the number of channels more significantly
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.identity_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.identity_conv(x)
        x = self.conv(x)
        x += identity
        x = self.relu(x)
        x = self.pool(x)
        return x

class VeryDeepUNet(nn.Module):
    def __init__(self):
        super(VeryDeepUNet, self).__init__()
        # Increasing the number of channels in each downsample block
        self.down1 = VeryDeepResidualDownsampleBlock(3, 128)
        self.down2 = VeryDeepResidualDownsampleBlock(128, 256)
        self.down3 = VeryDeepResidualDownsampleBlock(256, 512)
        self.down4 = VeryDeepResidualDownsampleBlock(512, 1024)

        # Middle block with extra layers
        self.middle = ExtraLayersBlock(1024, 1024)

        # Enhanced upsampling blocks
        self.up1 = AdvancedUpsampleBlock(1024, 512)
        self.up2 = AdvancedUpsampleBlock(512, 256)
        self.up3 = AdvancedUpsampleBlock(256, 128)
        self.up4 = AdvancedUpsampleBlock(128, 64)

        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.middle(x4)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x)

        x = self.out_conv(x)
        return x

model = VeryDeepUNet()


# Input: [1, 3, 256, 256]
# After Down1: [1, 64, 128, 128]
# After Down2: [1, 128, 64, 64]
# After Down3: [1, 256, 32, 32]
# After Down4: [1, 512, 16, 16]
# After Up1: [1, 256, 32, 32]
# After Up2: [1, 128, 64, 64]
# After Up3: [1, 64, 128, 128]
# Output: [1, 3, 128, 128]