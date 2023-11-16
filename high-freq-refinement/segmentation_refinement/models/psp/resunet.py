import torch
from torch import nn
from torch.nn import functional as F

from segmentation_refinement.models.psp import extractors

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
            Parameters for Deconvolution were chosen to avoid artifacts, following
            link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(middle_channels),
                nn.ELU(),
                nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU()
            )

    def forward(self, x):
        return self.block(x)


def cat_non_matching(x1, x2):
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]

    x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

    x = torch.cat([x1, x2], dim=1)
    return x

num_c = 2
class RefinementModule_Dual(nn.Module):
    def __init__(self, num_filters=32, is_deconv=False):
        super().__init__()
        self.feats = extractors.resnet152_dual()

        bottom_channel_nr = 2048
        self.pool = nn.MaxPool2d(2, 2)

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)


        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)

        self.final_28 = nn.Sequential(
            nn.Conv2d(num_filters * 8, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_c, kernel_size=1),
        )

        self.final_56 = nn.Sequential(
            nn.Conv2d(num_filters * 2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_c, kernel_size=1),
        )

        self.final_11 = nn.Conv2d(num_filters + 12, 32, kernel_size=1)

        self.final_21 = nn.Conv2d(32, num_c, kernel_size=1)

    def forward(self, x, seg, inter_s8=None, inter_s4=None):

        images = {}

        """
        First iteration, s8 output
        """
        if inter_s8 is None:
            p = torch.cat((x, seg, seg, seg), 1)

            _, _, conv4, conv5 = self.feats(p)
            pool = self.pool(conv5)
            center = self.center(pool)

            dec5 = self.dec5(cat_non_matching(conv5, center))
            dec4 = self.dec4(cat_non_matching(conv4, dec5))

            inter_s8 = self.final_28(dec4)
            r_inter_s8 = F.interpolate(inter_s8, scale_factor=8, mode='bilinear', align_corners=False)
            r_inter_tanh_s8 = torch.tanh(r_inter_s8)[:, :1].detach()
            r_inter_tanh_s8[torch.tanh(r_inter_s8)[:, :1] < torch.tanh(r_inter_s8)[:, 1:2]] = 0

            images['pred_28'] = torch.sigmoid(r_inter_s8)
            images['out_28'] = r_inter_s8
        else:
            if inter_s8.size()[1] > 1:
                r_inter_tanh_s8 = inter_s8[:, :1]
                r_inter_tanh_s8[inter_s8[:, :1] < inter_s8[:, 1:2]] = 0
            else:
                r_inter_tanh_s8 = inter_s8

        """
        Second iteration, s4 output
        """
        if inter_s4 is None:
            p = torch.cat((x, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)

            _, conv3, conv4, conv5 = self.feats(p)
            pool = self.pool(conv5)
            center = self.center(pool)

            dec5 = self.dec5(cat_non_matching(conv5, center))

            dec4 = self.dec4(cat_non_matching(conv4, dec5))
            dec3 = self.dec3(cat_non_matching(conv3, dec4))

            inter_s8_2 = self.final_28(dec4)
            r_inter_s8_2 = F.interpolate(inter_s8_2, scale_factor=8, mode='bilinear', align_corners=False)
            r_inter_tanh_s8_2 = torch.tanh(r_inter_s8_2)[:, :1].detach()
            r_inter_tanh_s8_2[torch.tanh(r_inter_s8_2)[:, :1] < torch.tanh(r_inter_s8_2)[:, 1:2]] = 0

            inter_s4 = self.final_56(dec3)
            r_inter_s4 = F.interpolate(inter_s4, scale_factor=4, mode='bilinear', align_corners=False)
            r_inter_tanh_s4 = torch.tanh(r_inter_s4)[:, :1].detach()
            r_inter_tanh_s4[torch.tanh(r_inter_s4)[:, :1] < torch.tanh(r_inter_s4)[:, 1:2]] = 0

            images['pred_28_2'] = torch.sigmoid(r_inter_s8_2)
            images['out_28_2'] = r_inter_s8_2
            images['pred_56'] = torch.sigmoid(r_inter_s4)
            images['out_56'] = r_inter_s4
        else:
            if inter_s8.size()[1] > 1:
                r_inter_tanh_s8_2 = inter_s8[:, :1]
                r_inter_tanh_s8_2[inter_s8[:, :1] < inter_s8[:, 1:2]] = 0
            else:
                r_inter_tanh_s8_2 = inter_s8

            if inter_s4.size()[1] > 1:
                r_inter_tanh_s4 = inter_s4[:, :1]
                r_inter_tanh_s4[inter_s4[:, :1] < inter_s4[:, 1:2]] = 0
            else:
                r_inter_tanh_s4 = inter_s4

        """
        Third iteration, s1 output
        """
        p = torch.cat((x, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)

        conv2, conv3, conv4, conv5 = self.feats(p)
        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(cat_non_matching(conv5, center))

        dec4 = self.dec4(cat_non_matching(conv4, dec5))
        dec3 = self.dec3(cat_non_matching(conv3, dec4))
        dec2 = self.dec2(cat_non_matching(conv2, dec3))
        dec1 = self.dec1(dec2)

        inter_s8_3 = self.final_28(dec4)
        r_inter_s8_3 = F.interpolate(inter_s8_3, scale_factor=8, mode='bilinear', align_corners=False)

        inter_s4_2 = self.final_56(dec3)
        r_inter_s4_2 = F.interpolate(inter_s4_2, scale_factor=4, mode='bilinear', align_corners=False)

        """
        Final output
        """
        p = F.relu(self.final_11(torch.cat([dec1, x], 1)), inplace=True)
        p = self.final_21(p)

        pred_224 = torch.sigmoid(p)

        images['pred_224'] = torch.sigmoid(r_inter_s4)[:, :1]
        images['out_224'] = p[:, :1]
        images['pred_28_3'] = torch.sigmoid(r_inter_s8_3)[:, :1]
        images['pred_56_2'] = torch.sigmoid(r_inter_s4_2)[:, :1]
        images['out_28_3'] = r_inter_s8_3[:, :1]
        images['out_56_2'] = r_inter_s4_2[:, :1]

        return images

