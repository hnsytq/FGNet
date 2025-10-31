import torch
import torch.nn as nn
from blocks import DiscrepancyModule, ResBlock

from einops import rearrange


class BandsFusion(nn.Module):
    def __init__(self, group_num, depth=4, base_features=16):
        super().__init__()

        self.group_num = group_num
        self.depth = depth
        in_dim, out_dim = group_num, base_features
        for idx in range(depth):
            self.add_module(f'stage_s_{idx}', nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                ResBlock(out_dim),
                #nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
            ))
            self.add_module(f'stage_r_{idx}', nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                ResBlock(out_dim),
                #nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
            ))
            self.add_module(f'cross_{idx}', DiscrepancyModule(out_dim, num_heads=8))
            in_dim = out_dim
            out_dim *= 2
        out_dim //= 2
        emb_dim = base_features * (2 ** (depth - 1))
        self.proj_conv = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dim // 2),
            nn.ReLU(),
            nn.Conv1d(emb_dim // 2, emb_dim, kernel_size=1, bias=False)
        )

        for idx in range(depth, 0, -1):
            if idx == depth:
                self.add_module(f'merge_{idx - 1}', ResBlock(out_dim))
                # out_dim *= 2
            else:
                self.add_module(f'unpool_{idx - 1}', nn.Sequential(
                    nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False),
                    #nn.UpsamplingBilinear2d(scale_factor=2)
                ))
                self.add_module(f'merge_{idx - 1}', nn.Sequential(
                    nn.Conv2d(2 * out_dim, out_dim, kernel_size=1, bias=False),
                    ResBlock(out_dim),
                    # nn.UpsamplingBilinear2d(scale_factor=2),
                    # nn.Conv2d(out_dim, out_dim // 2, kernel_size=1, bias=False)
                ))
            out_dim = out_dim // 2
        out_dim *= 2
        self.out_conv = nn.Sequential(
            #nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_dim, group_num, kernel_size=1, bias=False)
        )
        self.act = nn.ReLU()

    def forward_encoder(self, x):
        x_sel, x_res = torch.chunk(x, 2, dim=1)
        # out = x_sel
        fs, fs_seg = [], []
        for idx in range(self.depth):
            x_sel = self.__getattr__(f'stage_s_{idx}')(x_sel)
            x_res = self.__getattr__(f'stage_r_{idx}')(x_res)
            f = self.__getattr__(f'cross_{idx}')(x_res, x_sel)
            fs.append(f)
            fs_seg.append(x_sel + f)
        return fs, fs_seg

    def forward_decoder(self, fs):
        m = self.__getattr__(f'merge_{self.depth - 1}')(fs[self.depth - 1])
        for idx in range(self.depth - 1, 0, -1):
            m = torch.cat((fs[idx - 1], self.__getattr__(f'unpool_{idx - 1}')(m)), dim=1)
            m = self.__getattr__(f'merge_{idx - 1}')(m)
        return self.out_conv(m)

    def forward(self, x):
        x_sel, _ = torch.chunk(x, 2, dim=1)
        fs = self.forward_encoder(x)
        res = self.forward_decoder(fs)

        return self.act(x_sel + res)


if __name__ == '__main__':
    a = torch.randn((2, 30, 192, 192))
    c = torch.randn((2, 30, 192, 192))
    m = BandsFusion(15)
    b, s = m(a, c)
    print(b.shape)
