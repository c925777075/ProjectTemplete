from datetime import datetime
from copy import deepcopy

_base_ = ['../_base_/dataset/coco2014.py', '../_base_/dataset/coco2017.py']

now = datetime.now()
formatted_now = now.strftime("%Y_%m_%d")

training_args = dict(
    num_train_epochs=10,
    max_steps=20000,
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
