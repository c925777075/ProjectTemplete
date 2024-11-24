import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torchvision.models import VGG16_Weights
from src.models.vqvae.quantization import LFQ
from src.models.vqvae.discriminator import Discriminator, hinge_discr_loss, hinge_gen_loss, gradient_penalty
from torch.cuda.amp import autocast
from torch.autograd import grad as torch_grad
from src.models.root import MODELS
from transformers.modeling_utils import PreTrainedModel

@autocast(enabled = False)
def grad_layer_wrt_loss(
    loss: Tensor,
    layer: nn.Parameter
):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

class DWTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWTConv, self).__init__()
        self.conv = nn.Conv2d(4 * in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x
        x_ll = (x[..., ::2, ::2] + x[..., ::2, 1::2] + x[..., 1::2, ::2] + x[..., 1::2, 1::2]) * 0.5
        x_lh = (x[..., ::2, ::2] - x[..., ::2, 1::2] + x[..., 1::2, ::2] - x[..., 1::2, 1::2]) * 0.5
        x_hl = (x[..., ::2, ::2] - x[..., ::2, 1::2] + x[..., 1::2, ::2] - x[..., 1::2, 1::2]) * 0.5
        x_hh = (x[..., ::2, ::2] - x[..., ::2, 1::2] - x[..., 1::2, ::2] + x[..., 1::2, 1::2]) * 0.5
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)

        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=True)
        x = torch.cat([y, x], dim=1)
        x = self.conv2(x)
        return x

class DWTConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=1):
        super(DWTConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False)
        # self.conv1 = DWTConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.conv2 = DWTConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.relu2 = nn.ReLU()

        self.conv3 = nn.Identity()
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        self.conv4 = nn.Identity()
        if downsample > 1:
            self.conv4 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=(downsample*2-1, downsample*2-1),
                                   stride=(downsample, downsample),
                                   padding=downsample//2)

    def forward(self, x):
        identity = x  # 保存输入以进行跳跃连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.conv3(identity)

        out += identity  # 添加跳跃连接

        out = self.conv4(out)

        return out

class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super(VisionEncoder, self).__init__()
        self.conv_stem = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(64)

        self.blks = []
        self.blk_dims = [64] + cfg.backbone_base_dim
        for i, (layers, dim) in enumerate(zip(cfg.block, cfg.backbone_base_dim)):
            for j in range(layers):
                if j == 0:
                    input_dim = self.blk_dims[i]
                    output_dim = self.blk_dims[i+1]
                else:
                    input_dim = self.blk_dims[i+1]
                    output_dim = self.blk_dims[i+1]

                if j == layers-1:
                    self.blks.append(DWTConvBlock(input_dim, output_dim, downsample=2))
                else:
                    self.blks.append(DWTConvBlock(input_dim, output_dim))
        self.blks = nn.ModuleList(self.blks)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv_stem(x)
        x = self.act(x)
        x = self.norm(x)

        for fn in self.blks:
            x = fn(x)

        return x

class UpSamplerConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpSamplerConv, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=(3, 3), padding=1)
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(input_dim)

        self.conv2 = nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), padding=1)
        self.act2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(output_dim)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.act1(x)
        x = self.norm1(x)

        x = F.interpolate(x, (2*h, 2*w), mode="bilinear")
        x = self.conv2(x)
        x = self.act2(x)
        x = self.norm2(x)
        return x

class VisionDecoder(nn.Module):
    def __init__(self, cfg):
        super(VisionDecoder, self).__init__()
        self.blks = []
        self.blk_dims = cfg.backbone_base_dim + [64]
        for i, (layers, dim) in enumerate(zip(cfg.block, cfg.backbone_base_dim)):
            for j in range(layers):
                if j == 0:
                    input_dim = self.blk_dims[i]
                    output_dim = self.blk_dims[i + 1]
                else:
                    input_dim = self.blk_dims[i + 1]
                    output_dim = self.blk_dims[i + 1]

                if j == layers - 1:
                    self.blks.append(UpSamplerConv(input_dim, output_dim))
                else:
                    self.blks.append(UpSamplerConv(input_dim, output_dim))
        self.blks = nn.ModuleList(self.blks)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=(1, 1))

    def forward(self, x):
        for fn in self.blks:
            x = fn(x)

        x = F.interpolate(x, (2 * x.shape[2], 2 * x.shape[3]))
        x = self.conv_last(x)
        return x

@MODELS.register_module()
class VQVAEModel(nn.Module):
    def __init__(self, cfg):
        super(VQVAEModel, self).__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(cfg.VISION_ENCODER)
        self.quantizers = LFQ(**cfg.QUANTIZATION)

        self.vision_decoder = VisionDecoder(cfg.VISION_DECODER)
        if cfg.DISCRIMINATOR.use:
            self.discr = Discriminator(**cfg.DISCRIMINATOR)

        self.grad_penalty_loss_weight = 10
        self.quantizer_aux_loss_weight = 1.0
        self.perceptual_loss_weight = 0.1

        vgg_weights = VGG16_Weights.DEFAULT
        vgg = torchvision.models.vgg16(
            weights=vgg_weights
        )

        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        for name, para in vgg.named_parameters():
            para.requires_grad = False

        self.vgg = vgg
        self.use_gan = cfg.DISCRIMINATOR.use


    def forward(self, image, return_discr_loss=False, apply_gradient_penalty=True):
        x = image
        z = self.vision_encoder(x)
        z = z.unsqueeze(2)
        (quantized, codes, aux_losses), quantizer_loss_breakdown = self.quantizers(z, return_loss_breakdown=True)
        quantized = quantized.squeeze(2)
        rec_x = self.vision_decoder(quantized)
        recon_loss = F.mse_loss(rec_x, x)

        if return_discr_loss and self.use_gan:
            real = x
            fake = rec_x
            if apply_gradient_penalty:
                real = real.requires_grad_()
            real_logits = self.discr(real)
            fake_logits = self.discr(fake.detach())
            discr_loss = hinge_discr_loss(fake_logits, real_logits)
            gradient_penalty_loss = gradient_penalty(real, real_logits)
            total_loss = discr_loss + gradient_penalty_loss * self.grad_penalty_loss_weight
            return {"d_loss": total_loss}

        input_vgg_input = x
        recon_vgg_input = rec_x

        input_vgg_feats = self.vgg(input_vgg_input)
        recon_vgg_feats = self.vgg(recon_vgg_input)

        perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)

        if not self.use_gan:
            loss = recon_loss + aux_losses * self.quantizer_aux_loss_weight + perceptual_loss * self.perceptual_loss_weight
            return {"loss": loss, "reconstruct_loss": recon_loss, "aux_losses": aux_losses, "perceptual_loss": perceptual_loss}

        last_dec_layer = self.vision_decoder.conv_last.weight

        norm_grad_wrt_perceptual_loss = None
        if self.training:
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        fake_logits = self.discr(rec_x)
        gen_loss = hinge_gen_loss(fake_logits)

        adaptive_weight = 1.

        if norm_grad_wrt_perceptual_loss is not None:
            norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)
            adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-3)
            adaptive_weight.clamp_(max=1e3)

            if torch.isnan(adaptive_weight).any():
                adaptive_weight = 1.

        loss = recon_loss + aux_losses * self.quantizer_aux_loss_weight
        g_loss = perceptual_loss * self.perceptual_loss_weight + gen_loss * adaptive_weight
        return {"loss": loss,
                "g_loss": g_loss,
                "reconstruct_loss": recon_loss,
                "aux_losses": aux_losses,
                "perceptual_loss": perceptual_loss}