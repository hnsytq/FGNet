import torch
import torch.nn as nn
from einops import rearrange
import numbers
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_dim)
        self.norm2 = nn.BatchNorm2d(in_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(x + out)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def un_pool(input, scale):
    return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)


class AttenBlock(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(AttenBlock, self).__init__()
        self.num_heads = num_heads
        self.to_q = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        self.dw_q = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False)
        self.dw_k = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False)
        self.dw_v = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, bias=False)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        _, _, h, w = x.shape
        q = self.dw_q(self.to_q(x))
        k = self.dw_k(self.to_k(x))
        v = self.dw_v(self.to_v(x))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = entmax15(attn, dim=-1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.relu = nn.GELU()
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out = torch.cat((out1, out2), dim=1)
        return self.out_conv(self.relu(out))


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SelfAtten(nn.Module):
    def __init__(self, dim, heads, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                AttenBlock(dim, heads),
                LayerNorm(dim),
                # nn.Conv2d(dim, dim, kernel_size=1)
                PreNorm(dim, mult=4)
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (norm1, attn, norm2, ffn) in self.blocks:
            x = attn(norm1(x)) + x
            x = ffn(norm2(x)) + x
        return x


class MFCM(nn.Module):
    def __init__(self, out_dim, num_heads):
        super(MFCM, self).__init__()
        self.conv_x3 = nn.Conv2d(out_dim, out_dim, kernel_size=(3, 1), padding=(1, 0), groups=out_dim, bias=False)
        self.conv_x5 = nn.Conv2d(out_dim, out_dim, kernel_size=(5, 1), padding=(2, 0), groups=out_dim, bias=False)
        self.conv_x7 = nn.Conv2d(out_dim, out_dim, kernel_size=(7, 1), padding=(3, 0), groups=out_dim, bias=False)
        self.conv_y3 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, 3), padding=(0, 1), groups=out_dim, bias=False)
        self.conv_y5 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, 5), padding=(0, 2), groups=out_dim, bias=False)
        self.conv_y7 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, 7), padding=(0, 3), groups=out_dim, bias=False)

        self.att_x2y = CrossAtten(out_dim, num_heads)
        self.att_y2x = CrossAtten(out_dim, num_heads)
        self.project_out = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, fea):
        fea_x3 = self.conv_x3(fea)
        fea_x5 = self.conv_x5(fea)
        fea_x7 = self.conv_x7(fea)
        fea_y3 = self.conv_y3(fea)
        fea_y5 = self.conv_y5(fea)
        fea_y7 = self.conv_y7(fea)
        out_x = fea_x3 + fea_x5 + fea_x7
        out_y = fea_y3 + fea_y5 + fea_y7

        out_x1 = self.att_x2y(out_x, out_y)
        out_y1 = self.att_y2x(out_y, out_x)
        out = self.project_out(out_x1) + self.project_out(out_y1)
        return out


class UpsampleSeg(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim, num_heads):
        super(UpsampleSeg, self).__init__()
        self.num_heads = num_heads

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_dim1, out_dim, kernel_size=1, bias=False),
            ResBlock(out_dim)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_dim2, out_dim, kernel_size=1, bias=False),
            ResBlock(out_dim)
        )
        self.conv_dim = nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False)
        self.mfcm = MFCM(out_dim, num_heads)

    def forward(self, *xs):
        low, high = xs
        fea_1 = self.conv_1(low)
        fea_2 = self.conv_2(high)
        fea = torch.cat((fea_1, fea_2), dim=1)
        fea = self.conv_dim(fea)
        fea = un_pool(fea, 2)

        out = self.mfcm(fea)
        return out


class DiscrepancyModule(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        self.dw_q = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.dw_k = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.dw_v = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1_spa = LayerNorm(in_dim)
        self.norm1_spe = LayerNorm(in_dim)
        self.norm2 = LayerNorm(in_dim)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.ffn = PreNorm(in_dim)

    def forward(self, *xs):
        _, _, h, w = xs[0].shape
        fea_spatial = self.norm1_spa(xs[0])
        fea_spectral = self.norm1_spe(xs[1])

        q = self.dw_q(self.to_q(fea_spectral))
        k = self.dw_k(self.to_k(fea_spatial))
        v = self.dw_v(self.to_v(fea_spatial))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = 1 - attn

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.norm2(out)

        return self.ffn(out)


class CrossAtten(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        self.dw_q = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.dw_k = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.dw_v = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1_spa = LayerNorm(in_dim)
        self.norm1_spe = LayerNorm(in_dim)
        self.norm2 = LayerNorm(in_dim)
        self.norm_out = LayerNorm(in_dim)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.ffn = PreNorm(in_dim)

    def forward(self, *xs):
        _, _, h, w = xs[0].shape
        fea_spatial = self.norm1_spa(xs[0])
        fea_spectral = self.norm1_spe(xs[1])

        q = self.dw_q(self.to_q(fea_spectral))
        k = self.dw_k(self.to_k(fea_spatial))
        v = self.dw_v(self.to_v(fea_spatial))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.norm2(out)

        out = fea_spatial + out
        # return self.ffn(out + fea_spectral)
        return self.ffn(self.norm_out(out)) + out


class LGAM(nn.Module):
    def __init__(self, in_dim, rank_dim=15):
        super(LGAM, self).__init__()
        self.res = ResBlock(in_dim)

        self.lg_atten = LowRankAtten(in_dim, rank_dim=rank_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim)
        )
        self.act = nn.ReLU()

    def forward(self, x, lowrank):
        out_res = self.res(x)
        out_lg, lowrank = self.lg_atten(out_res, lowrank)
        out = x + out_lg
        return self.act(self.conv(out)), lowrank


class LowrankGen(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_dim = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.lowrank = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim // 2),
            nn.Linear(out_dim // 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: [b group_num group_num]
        x = self.conv_dim(x)  # b out_dim group_num
        return self.lowrank(x)


class LowRankAtten(nn.Module):
    def __init__(self, in_dim, rank_dim):
        super().__init__()

        self.relu = nn.ReLU()
        self.norm1 = LayerNorm(in_dim)
        self.lk = LowrankGen(rank_dim, in_dim)
        self.to_v = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim)
        )

        self.ffn = PreNorm(in_dim)
        self.norm2 = LayerNorm(in_dim)

    def forward(self, x, lowrank):

        b, c, h, w = x.shape
        x_norm = self.norm1(x)
        # x_reshape = rearrange(x, 'b c h w -> b c (h w)')
        v = self.to_v(x_norm)
        v = rearrange(v, 'b c h w -> b c (h w)')
        lowrank = self.lk(lowrank)

        attn = lowrank.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b c (h w) -> b c h w', h=h)
        out = out + x
        out_norm = self.norm2(out)
        # V = self.sc(spa_cof)

        return self.ffn(out_norm) + out, lowrank

