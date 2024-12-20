import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from src.dataset.root import DATASETS
from src.dataset.video_llm import video_transforms

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
class VideoDatasets(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.datasets = self.cfg.dataset
        self.data_path = []
        for dataset in self.datasets:
            dataset_name = dataset.get("name")
            dataset_path = dataset.get("path")
            self.data_path.append(dataset_path)

        data = []
        for data_path in self.data_path:
            files = os.listdir(data_path)
            for fil in files:
                data.append(os.path.join(data_path, fil))
        if mode == 'train':
            self.data = data[:int(len(data) * 0.98)]
        else:
            self.data = data[int(len(data) * 0.98):]
        self.transform = get_transforms_video(name="resize_crop", image_size=(cfg.DATA.SIZE[0], cfg.DATA.SIZE[1]))
        self.num_frames, self.frame_interval = cfg.DATA.NUM_FRAMES, cfg.DATA.FRAME_INTERVAL

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        video_file = self.data[item]
        # loading
        vframes, _, _ = torchvision.io.read_video(filename=video_file, pts_unit="sec", output_format="TCHW")
        cur_num_frame, C, H, W = vframes.shape
        if vframes.shape[0] < self.num_frames:
            pad = torch.zeros(self.num_frames - vframes.shape[0], C, H, W).type(vframes.dtype)
            vframes = torch.cat([vframes, pad], dim=0)
        video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
        video = self.transform(video).permute(1, 0, 2, 3)

        return video

    def collator(self, data):
        x = []
        for v in data:
            x.append(v)
        if len(data[0].shape) == 4:
            x = torch.stack(x, dim=0)
        elif len(data[0].shape) == 5:
            x = torch.cat(x, dim=0)
        return {"video": x}