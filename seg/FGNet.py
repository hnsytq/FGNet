import torch
import torch.nn as nn

from hamburger.ham import get_hams
from bands_fusion import BandsFusion

from blocks import LGAM, SelfAtten, UpsampleSeg, ResBlock


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Softmax(dim=1)
        super().__init__(conv2d, upsampling, activation)


class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(BottleNeck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            ResBlock(out_dim)
        )
        self.out = SelfAtten(out_dim, num_heads, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        return x


class DecoderSeg(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DecoderSeg, self).__init__()

        self.bottleneck = BottleNeck(in_dims[-1], out_dims[-1], num_heads=8)

        for i in range(4):
            t_dim = out_dims[-1]
            if i + 1 < 4:
                t_dim = out_dims[i + 1]
            self.add_module(f'de_{i + 1}', UpsampleSeg(in_dims[i], t_dim, out_dims[i], num_heads=8))

    def forward(self, xs):
        bottleneck = self.bottleneck(xs[-1])

        out_4 = self.de_4(xs[-1], bottleneck)
        out_3 = self.__getattr__('de_3')(xs[2], out_4)
        out_2 = self.__getattr__('de_2')(xs[1], out_3)
        out_1 = self.__getattr__('de_1')(xs[0], out_2)
        return out_1


class EncoderSeg(nn.Module):
    def __init__(self, in_dims, base_feature, group_num, depth=4):
        super().__init__()
        self.depth = depth

        out_dim = base_feature
        rank_dim = group_num
        for idx in range(depth):
            self.add_module(f'conv_{idx}', nn.Conv2d(in_channels=in_dims[idx],
                                                     out_channels=out_dim, kernel_size=3, padding=1))
            self.add_module(f'lga_{idx}', LGAM(out_dim, rank_dim))
            rank_dim = out_dim
            out_dim = out_dim * 2

    def forward(self, fs, lowrank):
        stages = []
        for idx in range(self.depth):
            # print(idx)
            x = self.__getattr__(f'conv_{idx}')(fs[idx])
            x, lowrank = self.__getattr__(f'lga_{idx}')(x, lowrank)
            stages.append(x)

        return stages


class FGNet(nn.Module):
    def __init__(self, group_num, base_feature, ham_args, classes=1):
        super().__init__()

        self.bands_fusion = BandsFusion(group_num)

        emb_dim = base_feature // 4
        in_dims = [emb_dim * (2 ** i) for i in range(4)]
        self.encoder_seg = EncoderSeg(in_dims, base_feature, group_num)

        out_dims = [base_feature, base_feature, base_feature * 2, base_feature * 2]
        self.decoder = DecoderSeg([base_feature * (2 ** i) for i in range(4)], out_dims)
        ham_type = getattr(ham_args, 'HAM_TYPE', 'NMF')
        HAM = get_hams(ham_type)
        self.ham = HAM(ham_args)
        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(
                nn.Conv2d(group_num, group_num, kernel_size=1, bias=False),
                nn.ReLU()
            )
        else:
            self.lower_bread = nn.Conv2d(group_num, group_num, kernel_size=1, bias=False)
        self.pre = SegmentationHead(
            in_channels=64,
            out_channels=classes,
            activation='sigmoid',
            kernel_size=1,
            upsampling=1,
        )

    def forward(self, x):

        x_sel = x[:, :x.shape[1] // 2, :, :]
        _, u = self.ham(self.lower_bread(x_sel))
        u = torch.squeeze(u, dim=1)

        fs_fusion, fs_seg = self.bands_fusion.forward_encoder(x)

        fs_seg = self.encoder_seg(fs_seg, u)

        out = self.decoder(fs_seg)
        out = self.pre(out)

        fuse = self.bands_fusion.forward_decoder(fs_fusion) + x_sel

        return fuse, out
