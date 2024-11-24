# 通用训练框架

## 简介

这个项目是一个通用的训练框架，旨在为机器学习和深度学习任务提供灵活的基础设施。该框架支持自定义数据格式、自定义模型结构以及易于使用的训练器（Trainer），使得模型的训练和评估变得更加高效和便捷。

## 特性

- **自定义数据格式**：支持多种输入数据格式，用户可以根据需求自定义数据加载和预处理流程。
- **自定义模型结构**：轻松定义和构建自定义模型，以适应不同的任务需求。
- **Trainer**：提供一个易于使用的训练器，支持分布式训练、模型检查点、日志记录和评估等功能。
- **集成支持**：与流行的深度学习库（如 PyTorch 和 TensorFlow）无缝集成。

## 安装

```bash
git clone https://github.com/c925777075/ProjectTemplete.git
cd ProjectTemplete
pip install -r requirements.txt
```

## config
```python
# 主要参考了mmengine的方法
# ./config/vqvae/vqvae.py
from datetime import datetime
from copy import deepcopy

_base_ = ['../_base_/dataset/coco2014.py', '../_base_/dataset/coco2017.py']

now = datetime.now()
formatted_now = now.strftime("%Y_%m_%d")

training_args = dict(
    num_train_epochs=5,
    max_steps=10000,
    do_train=True,
    do_eval=True,
    do_predict=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    lr_scheduler_type="constant_with_warmup",  # constant_with_warmup, linear
    learning_rate=1e-4,
    weight_decay=1e-6,
    warmup_ratio=0.05,
    save_strategy="steps",
    save_steps=1000,
    seed=42,
    bf16=True,
    fp16=False,
    dataloader_num_workers=8,
    remove_unused_columns=False,
    label_names=None,
    ddp_find_unused_parameters=True,
    dataloader_persistent_workers=True,
    resume_from_checkpoint=None,   # None   "/mnt/i/myai/MyLab/ProjectTemplete/exp/test_2024_11_24/checkpoint-120"
    ddp_timeout=1800,
    report_to="wandb",
    logging_steps=10,
    overwrite_output_dir=True,
    output_dir=f'./exp/test_{formatted_now}',
)

model_args = dict(
    type='VQVAEModel',
    cfg=dict(
        VISION_ENCODER=dict(
            backbone_base_dim=[128, 256, 512, 512],
            block=[1, 1, 1, 1],
        ),
        QUANTIZATION=dict(
            dim=512,
            codebook_size=1024,
            num_codebooks=1,
            entropy_loss_weight=0.1,
            commitment_loss_weight=1.0,
            diversity_gamma=2.5,
            soft_clamp_input_value=10,
            spherical=False,
        ),
        VISION_DECODER=dict(
            backbone_base_dim=[ 512, 512, 256, 128 ],
            block=[ 1, 1, 1, 1 ],
        ),
        DISCRIMINATOR=dict(
            use=False,
            dim=64,
            image_size=224,
            channels=3,
            max_dim=128,
        ),
    )
)

attribute_caption_coco_2014 = deepcopy(_base_.COCO2014.attribute_caption_coco2014)
attribute_caption_coco_2017 = deepcopy(_base_.COCO2017.attribute_caption_coco2017)

data_args = dict(
    train=dict(
        type='COCODateset',
        cfg=dict(
            DATA=dict(
                SIZE=224,
                DATA_ROOT="/mnt/i/data/coco2014",
                # DATA_ROOT=r"I:\data\coco2014",
            )
        ),
        mode="train",
    ),

    valid=dict(
        type='COCODateset',
        cfg=dict(
            DATA=dict(
                SIZE=224,
                DATA_ROOT="/mnt/i/data/coco2014",
                # DATA_ROOT=r"I:\data\coco2014",
            )
        ),
        mode="valid",
    ),
    test=None,
)

```

## 数据示例

```python
# src/dataset/vqvae/coco_datasets.py
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.dataset.root import DATASETS

@DATASETS.register_module()
class COCODateset(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        if mode == "valid":
            mode = "val"
        self.mode = mode
        img_dir = os.path.join(cfg.DATA.DATA_ROOT, f"{mode}2014_224")
        self.img_paths = [os.path.join(img_dir, p) for p in os.listdir(img_dir)]
        self.transform = transforms.Compose([
            transforms.Resize(cfg.DATA.SIZE),  # 将图像的短边缩放为224
            transforms.CenterCrop(cfg.DATA.SIZE),  # 中心裁剪为224x224
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
        ])
        self.num_samples = len(self.img_paths)
        print(f"{self.mode} datasets nums: {self.num_samples}")

    def __len__(self):
        if self.mode == "val":
            return min(self.num_samples, 100)
        return self.num_samples

    def __getitem__(self, item):
        image_path = self.img_paths[item]
        img = Image.open(image_path)
        img = img.convert("RGB")
        processed_img = self.transform(img)
        return {"image": processed_img}

    def collator(self, batch):
        images = []
        for f in batch:
            images.append(f['image'])
        images = torch.stack(images, dim=0)
        return {"image": images}
```

## 模型示例
```python
# src/models/vqvae/models.py
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
```
