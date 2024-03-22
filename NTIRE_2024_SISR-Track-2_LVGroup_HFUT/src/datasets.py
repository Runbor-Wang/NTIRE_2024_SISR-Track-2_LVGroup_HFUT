import os
import glob
import torch
import random

import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PairedImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=256):
        assert mode in ['train', ]
        self.data_source = data_source
        self.crop = crop
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset_refresh()

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index % len(self.img_paths)]).convert('RGB')
        gt = Image.open(self.gt_paths[index % len(self.gt_paths)]).convert('RGB')

        # crop
        width, height = img.size
        offset_w = random.randint(0, max(0, width - self.crop - 1))
        offset_h = random.randint(0, max(0, height - self.crop - 1))

        img = img.crop((offset_w, offset_h, offset_w + self.crop, offset_h + self.crop))
        gt = gt.crop((offset_w * 4, offset_h * 4, offset_w * 4 + self.crop * 4, offset_h * 4 + self.crop * 4))

        # horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            
        img = self.transform(img)
        gt = self.transform(gt)

        return img, gt

    def __len__(self):
        return max(len(self.img_paths), len(self.gt_paths))
        
    def dataset_refresh(self):
        self.img_paths = sorted(glob.glob(self.data_source + '/Train/LR_x4/*.*'))
        # print('length of the self.img_paths is :', len(self.img_paths))
        self.gt_paths = sorted(glob.glob(self.data_source + '/Train/HR/*.*'))


class SingleImgDataset(Dataset):
    def __init__(self, data_source, mode):
        assert mode in ['val', 'test']
        self.data_source = data_source
        
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.dataset_refresh()

    def __getitem__(self, index):
                              
        path = self.img_paths[index % len(self.img_paths)]
        
        name = os.path.basename(path)
        
        img = Image.open(path).convert('RGB')
        
        img = self.transform(img)
        
        return img, name

    def __len__(self):
        return len(self.img_paths)

    def dataset_refresh(self):

        # self.img_paths = sorted(glob.glob(data_source + '/' + 'test' + '/blurry' + '/*/*.*'))
        self.img_paths = sorted(glob.glob(self.data_source + '/LR_x4/*.*'))
        # self.img_paths = sorted(glob.glob(self.data_source + '/*.*'))
