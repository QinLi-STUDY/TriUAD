import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
            self.img_segment_path = os.path.join(root, 'train_segment')
            self.img_sketch_path = os.path.join(root, 'train_sketch')
        else:
            self.img_path = os.path.join(root, 'test')
            self.img_segment_path = os.path.join(root, 'test_segment')
            self.img_sketch_path = os.path.join(root, 'test_sketch')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types, self.segment_paths, self.sketch_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        tot_segment_paths = []
        tot_sketch_paths = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                            
                img_segment_paths = glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.bmp")
                img_sketch_paths = glob.glob(os.path.join(self.img_sketch_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_sketch_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_sketch_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_sketch_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                # print(img_paths)
                # print(img_segment_paths)
                # print(img_sketch_paths)
                
                tot_segment_paths.extend(img_segment_paths)
                tot_sketch_paths.extend(img_sketch_paths)
                
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                #glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.JPG") + \
                img_segment_paths = glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_segment_path, defect_type) + "/*.bmp")
                img_sketch_paths = glob.glob(os.path.join(self.img_sketch_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_sketch_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_sketch_path, defect_type) + "/*.bmp")
                
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_segment_paths.sort()
                img_sketch_paths.sort()
                
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                
                tot_segment_paths.extend(img_segment_paths)
                tot_sketch_paths.extend(img_sketch_paths)
                
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types), np.array(tot_segment_paths), np.array(tot_sketch_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type, segment_path, sketch_path = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx], self.segment_paths[idx], self.sketch_paths[idx]
        
        # print("img_path", img_path)
        # print("segment_path", segment_path)
        # print("sketch_path", sketch_path)
        # print("gt", gt)
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        segment = Image.open(segment_path).convert('RGB')
        segment = self.transform(segment)
        sketch = Image.open(sketch_path).convert('RGB')
        sketch = self.transform(sketch)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        

        return img, gt, label, img_path, segment, sketch


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase
        ########################################     
        self.img_seg_path = os.path.join(root, 'realiad_1024_seg', category)
        self.img_sketch_path = os.path.join(root, 'realiad_1024_sketch', category)
        self.img_seg_paths, self.img_sketch_paths = [], []

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []
        
        

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])
            ############
            self.img_seg_paths.append(os.path.join(root, 'realiad_1024_seg', category, sample['image_path']))
            self.img_sketch_paths.append(os.path.join(root, 'realiad_1024_sketch', category, sample['image_path']))

        self.img_paths = np.array(self.img_paths)
        ###########################
        self.img_seg_paths = np.array(self.img_seg_paths)
        self.img_sketch_paths = np.array(self.img_sketch_paths)
        ###########################
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        img_seg_path, img_sketch_path = self.img_seg_paths[idx],self.img_sketch_paths[idx]
        img_seg = Image.open(img_seg_path).convert('RGB')
        img_ske = Image.open(img_sketch_path).convert('RGB')
        img_seg = self.transform(img_seg)
        img_ske = self.transform(img_ske)
        

        if self.phase == 'train':
            return img,label,img_seg,img_ske

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, img_seg, img_ske



