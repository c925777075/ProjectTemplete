import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.dataset.root import DATASETS
from src.dataset.video_llm import video_transforms
from transformers import AutoTokenizer

def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video

def temporal_random_crop(vframes, num_frames, frame_interval):
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert end_frame_ind - start_frame_ind >= num_frames
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indice]
    return video


@DATASETS.register_module()
class VideoDatasetsCondition(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.datasets = self.cfg.dataset
        self.data_path = []
        dataset_info = None
        for dataset in self.datasets:
            dataset_name = dataset.get("name")
            dataset_path = dataset.get("path")
            dataset_info = dataset.get ("info")
            self.data_path.append(dataset_path)

        df = pd.read_csv(dataset_info)
        mapping = dict(zip(df['video'], df['caption']))
        mapping_frame = dict(zip(df['video'], df['frame']))
        mapping_fps = dict(zip(df['video'], df['fps']))

        data = []
        for data_path in self.data_path:
            files = os.listdir(data_path)
            for fil in files:
                if fil in mapping and fil in mapping_frame and fil in mapping_fps:
                    num_frame = mapping_frame[fil]
                    fps = mapping_fps[fil]
                    if num_frame <= 10 * fps:
                        video_path = os.path.join(data_path, fil)
                        data.append({"file_name": fil, "caption": mapping[fil], "mp4": video_path})
        

        if mode == 'train':
            self.data = data[:int(len(data) * 0.98)]
        else:
            self.data = data[int(len(data) * 0.98):]
        self.transform = get_transforms_video(name="resize_crop", image_size=(cfg.DATA.SIZE[0], cfg.DATA.SIZE[1]))
        self.num_frames, self.frame_interval = cfg.DATA.NUM_FRAMES, cfg.DATA.FRAME_INTERVAL

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        video_info = self.data[item]
        video_file = video_info['mp4']
        video_caption = video_info['caption']
        # loading
        vframes, _, _ = torchvision.io.read_video(filename=video_file, pts_unit="sec", output_format="TCHW")
        cur_num_frame, C, H, W = vframes.shape
        if vframes.shape[0] < self.num_frames:
            pad = torch.zeros(self.num_frames - vframes.shape[0], C, H, W).type(vframes.dtype)
            vframes = torch.cat([vframes, pad], dim=0)
        video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
        video = self.transform(video).permute(1, 0, 2, 3)

        return {"video": video, "caption": video_caption}

    def collator(self, data):
        captions = []
        x = []
        for d in data:
            x.append(d['video'])
            captions.append(d['caption'])

        x = torch.stack(x, dim=0)
        
        text_inputs = self.tokenizer(
            captions,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        return {"video": x, "text_inputs": text_inputs}