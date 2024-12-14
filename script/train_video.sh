# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
export USE_SACRED="false"
export WANDB_DISABLED="false"
export WANDB_PROJECT="videolm"  # name your W&B project

python src/pipeline/train.py --config config/videolm/videolm.py
# nohup accelerate launch --config_file default_config.yaml src/pipeline/train.py > "$TRAIN_LOG_FILE" 2>&1 &
