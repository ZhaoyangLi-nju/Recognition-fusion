from PIL import Image
from PIL import ImageFile
# from torchvision.datasets.folder import find_classes
# from torchvision.datasets.folder import make_dataset
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os.path
import random
import torch
import util.utils as utils
from torchvision.transforms import functional as F
import copy
def find_classes(dir):
    # 得到指定目录下的所有文件，并将其名字和指定目录的路径合并
    # 以数组的形式存在classes中
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # 使用sort()进行简单的排序
    classes.sort()
    # 将其保存的路径排序后简单地映射到 0 ~ [ len(classes)-1] 的数字上
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # 返回存放路径的数组和存放其映射后的序号的数组
    return classes, class_to_idx
def make_dataset(dir, class_to_idx, extensions):
    images = []
    # expanduser把path中包含的"~"和"~user"转换成用户目录
    # 主要还是在Linux之类的系统中使用，在不包含"~"和"~user"时
    # dir不变
    dir = os.path.expanduser(dir)
    # 排序后按顺序通过for循环dir路径下的所有文件名
    for target in sorted(os.listdir(dir)):
        # 将路径拼合
        d = os.path.join(dir, target)
        # 如果拼接后不是文件目录，则跳出这次循环
        if not os.path.isdir(d):
            continue
        # os.walk(d) 返回的fnames是当前d目录下所有的文件名
        # 注意：第一个for其实就只循环一次，返回的fnames 是一个数组
        for root, _, fnames in sorted(os.walk(d)):
            # 循环每一个文件名
            for fname in sorted(fnames):
                # 文件的后缀名是否符合给定
                # if has_file_allowed_extension(fname, extensions):
                    # 组合路径
                path = os.path.join(root, fname)
                # 将组合后的路径和该文件位于哪一个序号的文件夹下的序号
                # 组成元祖
                item = (path, class_to_idx[target])
                # 将其存入数组中
                images.append(item)

    return images
class SingleDataset:

    def __init__(self, cfg, data_dir=None, transform=None, labeled=True):
        self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = labeled

        self.classes, self.class_to_idx = find_classes(self.data_dir)
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.imgs = make_dataset(self.data_dir, self.class_to_idx, ['jpg','png'])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img,label
class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        A= sample

        # if self.padding > 0:
        #     A = F.pad(A, self.padding)


        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))

        i, j, h, w = self.get_params(A, self.size)
        sample= F.crop(A, i, j, h, w)


        # _i, _j, _h, _w = self.get_params(A, self.size)
        # sample['A'] = F.crop(A, i, j, h, w)
        # sample['B'] = F.crop(B, _i, _j, _h, _w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        A= sample
        sample= F.center_crop(A, self.size)

        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        A= sample
        if random.random() > 0.5:
            A = F.hflip(A)


        sample= A


        return sample


class Resize(transforms.Resize):

    def __call__(self, sample):
        A = sample
        h = self.size[0]
        w = self.size[1]

        sample= F.resize(A, (h, w))


        return sample


class ToTensor(object):
    def __call__(self, sample):

        A= sample

        # if isinstance(sample, dict):
        #     for key, value in sample:
        #         _list = sample[key]
        #         sample[key] = [F.to_tensor(item) for item in _list]

        sample= F.to_tensor(A)


        return sample

class Normalize(transforms.Normalize):

    def __call__(self, sample):
        A = sample
        sample = F.normalize(A, self.mean, self.std)

        return sample

class Lambda(transforms.Lambda):

    def __call__(self, sample):
        return self.lambd(sample)
