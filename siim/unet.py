import torch
from torch import nn
from torchvision.models import resnet50, resnet18, resnet34
from torch import einsum
import torch.nn.functional as F
# from resnet import resnet34
try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvPRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.PReLU(num_parameters=out, init=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            ConvRelu(in_channels, out_channels),
            ConvRelu(out_channels, out_channels),
            ConvRelu(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlockPrelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlockPrelu, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            ConvPRelu(in_channels, out_channels),
            ConvPRelu(out_channels, out_channels),
            ConvPRelu(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UnetOverResnet50(nn.Module):

    def __init__(self, num_up_filters=512, pretrained=False, ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = resnet50(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   encoder.relu)

        self.conv2 = encoder.layer1

        self.conv3 = encoder.layer2

        self.conv4 = encoder.layer3

        self.conv5 = encoder.layer4

        self.center = DecoderBlock(2048, num_up_filters)
        self.dec5 = DecoderBlock(2048 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlock(1024 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlock(512 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlock(256 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlock(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvRelu(num_up_filters // 32, num_up_filters // 32)
        self.final = nn.Conv2d(num_up_filters // 32, 1, kernel_size=1)

        self.up_sample = nn.functional.interpolate

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))

        x_out = self.final(dec0)

        return x_out


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class UnetOverResnet34(nn.Module):
    def __init__(self, num_up_filters=512, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = resnet34(pretrained=pretrained)

        self.prelu = nn.PReLU(num_parameters=64, init=0)

        self.conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   self.prelu)

        self.conv2 = encoder.layer1

        self.conv3 = encoder.layer2

        self.conv4 = encoder.layer3

        self.conv5 = encoder.layer4

        self.center = DecoderBlockPrelu(512, num_up_filters)
        self.dec5 = DecoderBlockPrelu(512 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlockPrelu(256 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlockPrelu(128 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlockPrelu(64 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlockPrelu(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvPRelu(num_up_filters // 32, num_up_filters // 32)
        self.final = nn.Conv2d(num_up_filters // 32, 1, kernel_size=1)

        self.up_sample = nn.functional.interpolate

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))

        x_out = self.final(dec0)

        return x_out


class UnetOverResnet18(nn.Module):
    def __init__(self, num_up_filters=512, pretrained=False, ):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = resnet18(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   encoder.relu)

        self.conv2 = encoder.layer1

        self.conv3 = encoder.layer2

        self.conv4 = encoder.layer3

        self.conv5 = encoder.layer4

        self.center = DecoderBlock(512, num_up_filters)
        self.dec5 = DecoderBlock(512 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlock(256 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlock(128 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlock(64 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlock(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvRelu(num_up_filters // 32, num_up_filters // 32)
        self.final = nn.Conv2d(num_up_filters // 32, 1, kernel_size=1)

        self.up_sample = nn.functional.interpolate

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        # dec1 = nn.Dropout2d(p=0.4)(dec1)

        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))

        x_out = self.final(dec0)

        return x_out


class UnetOverResnet34Class(nn.Module):

    def __init__(self, num_up_filters=512, pretrained=True, mode='train'):
        super().__init__()
        self.mode = mode
        self.pool = nn.MaxPool2d(2, 2)

        encoder = resnet34(pretrained=pretrained)

        self.relu = nn.PReLU(num_parameters=64, init=0)

        self.conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   self.relu)

        self.conv2 = encoder.layer1

        self.conv3 = encoder.layer2

        self.conv4 = encoder.layer3

        self.conv5 = encoder.layer4

        self.fuse_image = nn.Sequential(nn.Linear(512, 16),
                                        nn.PReLU(num_parameters=16, init=0))
        self.logit_image = nn.Sequential(nn.Linear(16, 1),
                                         nn.Sigmoid())

        self.dropout2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.4)

        self.center = DecoderBlockPrelu(512, num_up_filters)
        self.dec5 = DecoderBlockPrelu(512 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlockPrelu(256 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlockPrelu(128 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlockPrelu(64 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlockPrelu(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvPRelu(num_up_filters // 32, num_up_filters // 32)
        self.final = nn.Conv2d(num_up_filters // 32, 1, kernel_size=1)

        self.up_sample = nn.functional.interpolate

        self.logit = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
                                   nn.PReLU(num_parameters=16, init=0),
                                   nn.Conv2d(16, 1, kernel_size=1, bias=False))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        e = F.adaptive_avg_pool2d(conv5, output_size=1).view(conv5.size(0), -1)  # 512
        e = self.dropout(e)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        dec1 = self.dropout2d(dec1)

        # classification process
        fuse_image = self.fuse_image(e)  # 16
        logit_image = self.logit_image(fuse_image)  # 1

        # segmentation process
        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))
        x_out = self.final(dec0)

        # combine
        fuse = torch.cat([dec0,
                          F.upsample(fuse_image.view(fuse_image.size(0), -1, 1, 1), scale_factor=1024, mode='bilinear',
                                     align_corners=True)], 1)  # 32, 1024, 1024
        logit = self.logit(fuse)  # 1, 1024, 1024

        if self.mode == 'train':
            return logit, logit_image.view(-1), x_out

        else:
            return logit


class Unet34(nn.Module):
    def __init__(self, num_up_filters=512, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   self.relu)

        self.conv2 = encoder.layer1

        self.conv3 = encoder.layer2

        self.conv4 = encoder.layer3

        self.conv5 = encoder.layer4

        self.center = DecoderBlock(512, num_up_filters)
        self.dec5 = DecoderBlock(512 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlock(256 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlock(128 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlock(64 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlock(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvRelu(num_up_filters // 32, num_up_filters // 32)
        self.final = nn.Conv2d(num_up_filters // 32, 1, kernel_size=1)

        self.up_sample = nn.functional.interpolate

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))

        x_out = self.final(dec0)

        return x_out


class UnetOverResnet34SCSE(nn.Module):
    def __init__(self, num_up_filters=512, pretrained=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = resnet34(pretrained=pretrained)

        self.prelu = nn.PReLU(num_parameters=64, init=0)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   self.relu)

        self.conv2 = nn.Sequential(encoder.layer1,
                                   SCse(64))

        self.conv3 = nn.Sequential(encoder.layer2,
                                   SCse(128))

        self.conv4 = nn.Sequential(encoder.layer3,
                                   SCse(256))

        self.conv5 = nn.Sequential(encoder.layer4,
                                   SCse(512))

        self.center = DecoderBlock(512, num_up_filters)
        self.dec5 = DecoderBlock(512 + num_up_filters, num_up_filters // 2)
        self.dec4 = DecoderBlock(256 + num_up_filters // 2, num_up_filters // 4)
        self.dec3 = DecoderBlock(128 + num_up_filters // 4, num_up_filters // 8)
        self.dec2 = DecoderBlock(64 + num_up_filters // 8, num_up_filters // 16)
        self.dec1 = DecoderBlock(64 + num_up_filters // 16, num_up_filters // 32)
        self.dec0 = ConvRelu(num_up_filters // 32, num_up_filters // 32)
        self.final = nn.Conv2d(num_up_filters // 32, 1, kernel_size=1)

        self.up_sample = nn.functional.interpolate

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(conv1)
        conv2 = self.conv2(conv1_pool)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([self.up_sample(center, scale_factor=2), conv5], 1))
        dec4 = self.dec4(torch.cat([self.up_sample(dec5, scale_factor=2), conv4], 1))
        dec3 = self.dec3(torch.cat([self.up_sample(dec4, scale_factor=2), conv3], 1))
        dec2 = self.dec2(torch.cat([self.up_sample(dec3, scale_factor=2), conv2], 1))
        dec1 = self.dec1(torch.cat([self.up_sample(dec2, scale_factor=2), conv1], 1))

        dec0 = self.dec0(self.up_sample(dec1, scale_factor=2))

        x_out = self.final(dec0)

        return x_out
