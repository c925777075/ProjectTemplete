from datetime import datetime
from copy import deepcopy

_base_ = ['../_base_/dataset/video_general.py']

now = datetime.now()
formatted_now = now.strftime("%Y_%m_%d")

training_args = dict(
    num_train_epochs=10,
    max_steps=100000,
    do_train=True,
    do_eval=True,
    do_predict=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    lr_scheduler_type="cosine",  # constant_with_warmup, linear
    learning_rate=1e-4,
    weight_decay=1e-6,
    warmup_ratio=0.0,
    save_strategy="steps",
    save_steps=20000,
    seed=42,
    max_grad_norm=20,
    bf16=True,
    fp16=False,
    dataloader_num_workers=16,
    remove_unused_columns=False,
    label_names=None,
    ddp_find_unused_parameters=True,
    dataloader_persistent_workers=True,
    resume_from_checkpoint=None,   # None   "/mnt/i/myai/MyLab/ProjectTemplete/exp/test_2024_11_24/checkpoint-120"
    ddp_timeout=1800,
    report_to="none",  # None
    logging_steps=5,
    overwrite_output_dir=True,
    output_dir=f'./exp/test_{formatted_now}',
)

model_args = dict(
    type='VideoLLM',
    cfg=dict(
        VISION_ENCODER=dict(
            checkpoint_enc=f'/mnt/chenyu/pretrain_models/Cosmos-Tokenizer-DV4x8x8/encoder.jit'
        ),
        VISION_DECODER=dict(
            checkpoint_dec=f'/mnt/chenyu/pretrain_models/Cosmos-Tokenizer-DV4x8x8/decoder.jit'
        ),
        LLM=dict(
            enable=True,
            model_path="/mnt/chenyu/pretrain_models/Qwen2.5-0.5B-Instruct",
        ),
        MLLM=dict(
            vocab_size=64008,
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_act="silu",
            max_position_embeddings=8196,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            attn_implementation="eager",
        ),
        TIME_RATIO=4,  # Cosmos-Tokenizer-DV4x8x8
    )
)

attribute_video_general = deepcopy(_base_.VIDEO_GENERAL.attribute_video_general)
# attribute_caption_coco_2017 = deepcopy(_base_.COCO2017.attribute_video_test)

data_args = dict(
    train=dict(
        type='VideoDatasetsCondition',
        cfg=dict(
            dataset=[attribute_video_general],
            DATA=dict(
                SIZE=(120, 160),
                NUM_FRAMES=30,
                FRAME_INTERVAL=3,
            ),
            tokenizer="/mnt/chenyu/pretrain_models/Qwen2.5-0.5B-Instruct"
        ),
        mode="train",
    ),

    valid=dict(
        type='VideoDatasetsCondition',
        cfg=dict(
            dataset=[attribute_video_general],
            DATA=dict(
                SIZE=(120, 160),
                NUM_FRAMES=30,
                FRAME_INTERVAL=3,
            ),
            tokenizer="/mnt/chenyu/pretrain_models/Qwen2.5-0.5B-Instruct"
        ),
        mode="valid",
    ),
    test=None,
)
