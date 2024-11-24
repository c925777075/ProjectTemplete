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