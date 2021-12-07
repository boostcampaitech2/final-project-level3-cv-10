# built-in
import logging

# torch
import torch
import torch.nn as nn

logger = logging.getLogger('model')


class EncodeConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(EncodeConv2d, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.bn2d = nn.BatchNorm2d(num_features=out_channels)
        self.actFunc = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.actFunc(x)
        return x


class EncodeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(EncodeBlock, self).__init__()

        self.conv1 = EncodeConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
        )
        self.conv2 = EncodeConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def crop_img(in_tensor, out_size):
    dim_w, dim_h = in_tensor.size()[-2:]
    out_tensor = in_tensor[:, :,
                           int((dim_w - out_size) / 2):int((dim_w + out_size) /
                                                           2),
                           int((dim_h - out_size) / 2):int((dim_h + out_size) /
                                                           2), ]
    return out_tensor


class UNet(nn.Module):
    def __init__(self, classes):
        super(UNet, self).__init__()

        self.classes = classes
        self.num_classes = len(self.classes)

        self.encoder1 = EncodeBlock(3, 64)
        self.encoder2 = EncodeBlock(64, 128)
        self.encoder3 = EncodeBlock(128, 256)
        self.encoder4 = EncodeBlock(256, 512)

        self.decoder5 = nn.Sequential(
            EncodeBlock(512, 1024),
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=512,
                               kernel_size=2,
                               stride=2))
        self.decoder4 = nn.Sequential(
            EncodeBlock(1024, 512),
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=2,
                               stride=2))
        self.decoder3 = nn.Sequential(
            EncodeBlock(512, 256),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=2,
                               stride=2))
        self.decoder2 = nn.Sequential(
            EncodeBlock(256, 128),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=2,
                               stride=2))
        self.decoder1 = nn.Sequential(
            EncodeBlock(128, 64),
            nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1),
        )
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        enc1_conv_output = self.encoder1(input)  # 64
        enc1_output = self.maxpool2d(enc1_conv_output)

        enc2_conv_output = self.encoder2(enc1_output)  # 128
        enc2_output = self.maxpool2d(enc2_conv_output)

        enc3_conv_output = self.encoder3(enc2_output)  # 256
        enc3_output = self.maxpool2d(enc3_conv_output)

        enc4_conv_output = self.encoder4(enc3_output)  # 512
        enc4_output = self.maxpool2d(enc4_conv_output)

        dec5_output = self.decoder5(enc4_output)

        # skip4 + enc4 output
        # croped_enc4_output = crop_img(enc4_conv_output,
        #                               dec5_output.shape[-2])  # skip4
        croped_enc4_output = enc4_conv_output
        merge_enc4_output = torch.cat([croped_enc4_output, dec5_output], dim=1)
        dec4_output = self.decoder4(merge_enc4_output)

        # skip3 + enc3 output
        # croped_enc3_output = crop_img(enc3_conv_output,
        #                               dec4_output.shape[-2])  # skip3
        croped_enc3_output = enc3_conv_output
        merge_enc3_output = torch.cat([croped_enc3_output, dec4_output], dim=1)
        dec3_output = self.decoder3(merge_enc3_output)

        # skip2 + enc2 output
        # croped_enc2_output = crop_img(enc2_conv_output,
        #                               dec3_output.shape[-2])  # skip2
        croped_enc2_output = enc2_conv_output
        merge_enc2_output = torch.cat([croped_enc2_output, dec3_output], dim=1)
        dec2_output = self.decoder2(merge_enc2_output)

        # skip1 + enc1 output
        # croped_enc1_output = crop_img(enc1_conv_output,
        #                               dec2_output.shape[-2])  # skip1
        croped_enc1_output = enc1_conv_output
        merge_enc1_output = torch.cat([croped_enc1_output, dec2_output], dim=1)
        dec1_output = self.decoder1(merge_enc1_output)

        return dec1_output
