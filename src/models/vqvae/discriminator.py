import torch
import torch.nn.functional as F
from math import log2
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from torch import nn, einsum, Tensor
from kornia.filters import filter3d
from torch.autograd import grad as torch_grad
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from functools import wraps, partial


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def safe_get_index(it, ind, default = None):
    if ind < len(it):
        return it[ind]
    return default

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))

def is_odd(n):
    return not divisible_by(n, 2)

def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

# gan related

def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = 1)
        return F.gelu(gate) * x

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(
        self,
        x,
        space_only = False,
        time_only = False
    ):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum('i, j -> i j', f, f)
            f = rearrange(f, '... -> 1 1 ...')
        elif time_only:
            f = rearrange(f, 'f -> 1 f 1 1')
        else:
            f = einsum('i, j, k -> i j k', f, f, f)
            f = rearrange(f, '... -> 1 ...')

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        out = filter3d(x, f, normalized = True)

        if is_images:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

class AdaptiveRMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_cond,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.dim_cond = dim_cond
        self.channel_first = channel_first
        self.scale = dim ** 0.5

        self.to_gamma = nn.Linear(dim_cond, dim)
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    def forward(self, x: Tensor, *, cond: Tensor):
        batch = x.shape[0]
        assert cond.shape == (batch, self.dim_cond)

        gamma = self.to_gamma(cond)

        bias = 0.
        if exists(self.to_bias):
            bias = self.to_bias(cond)

        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)

            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)

        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * gamma + bias

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class LinearAttention(nn.Module):
    """
    using the specific linear attention proposed in https://arxiv.org/abs/2106.09681
    """

    def __init__(
        self,
        *,
        dim,
        dim_cond = None,
        dim_head = 8,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.need_cond = exists(dim_cond)

        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond = dim_cond)
        else:
            self.norm = RMSNorm(dim)

        self.attn = TaylorSeriesLinearAttn(
            dim = dim,
            dim_head = dim_head,
            heads = heads
        )

    def forward(
        self,
        x,
        cond = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if self.need_cond else dict()

        x = self.norm(x, **maybe_cond_kwargs)

        return self.attn(x)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LinearSpaceAttention(LinearAttention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c ... h w -> b ... h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b ... h w c -> b c ... h w')

class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True,
        antialiased_downsample = True
    ):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding = 1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding = 1),
            leaky_relu()
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only = True)

            x = self.downsample(x)

        x = (x + res) * (2 ** -0.5)
        return x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_cond = None,
        mult = 4,
        images = False
    ):
        super().__init__()
        conv_klass = nn.Conv2d if images else nn.Conv3d

        rmsnorm_klass = RMSNorm if not exists(dim_cond) else partial(AdaptiveRMSNorm, dim_cond = dim_cond)

        maybe_adaptive_norm_klass = partial(rmsnorm_klass, channel_first = True, images = images)

        dim_inner = int(dim * mult * 2 / 3)

        self.norm = maybe_adaptive_norm_klass(dim)

        self.net = nn.Sequential(
            conv_klass(dim, dim_inner * 2, 1),
            GEGLU(),
            conv_klass(dim_inner, dim, 1)
        )

    def forward(
        self,
        x: Tensor,
        *,
        cond = None
    ):
        maybe_cond_kwargs = dict(cond = cond) if exists(cond) else dict()

        x = self.norm(x, **maybe_cond_kwargs)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        max_dim = 512,
        attn_heads = 8,
        attn_dim_head = 32,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        ff_mult = 4,
        antialiased_downsample = False,
        use = None,
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample = is_not_last,
                antialiased_downsample = antialiased_downsample
            )

            attn_block = nn.Sequential(
                Residual(LinearSpaceAttention(
                    dim = out_chan,
                    heads = linear_attn_heads,
                    dim_head = linear_attn_dim_head
                )),
                Residual(FeedForward(
                    dim = out_chan,
                    mult = ff_mult,
                    images = True
                ))
            )

            blocks.append(nn.ModuleList([
                block,
                attn_block
            ]))

            image_resolution //= 2

        self.blocks = nn.ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):

        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)