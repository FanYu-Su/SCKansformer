from PIL import Image
from matplotlib.cbook import ls_mapper
import torch
from torch.utils.data import Dataset
import random
import os

data_name_dic = ['Haemocytoblast', 'Myeloblast', 'Promyelocyte', 'Neutrophilic myelocyte', 'Neutrophilic metamyelocyte', 'Neutrophilic granulocyte band form', 'Neutrophilic granulocyte segmented form', 'Acidophil in young', 'Acidophil late young', 'Acidophillic rod-shaped nucleus', 'Eosinophillic phloem granulocyte', 'Basophillic in young', 'Basophillic late young', 'Basophillic rod-shaped nucleus',
                 'Basophllic lobule nucleus', 'Pronormoblast', 'Prorubricyte', 'Polychromatic erythroblast', 'Metarubricyte', 'Prolymphocyte', 'Mature lymphocyte', 'Hetertypic lymphocyte', 'Primitive monocyte', 'Promonocyte', 'Mature monocyte', 'Plasmablast', 'infantile plasmocyte', 'Matrue plasmocyte', 'Bistiocyte', 'Juvenile cell', 'Granulocyte megakaryocyte', 'Naked megakaryocyte']


class Five_Flowers_Load(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform

        random.seed(0)  # 保证随机结果可复现
        assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(os.path.join(data_path))]
        self.num_class = len(flower_class)
        # 排序，保证顺序一致
        flower_class.sort()

        class_indices = dict((cla, idx) for idx, cla in enumerate(flower_class))
        # 字典，构成一个类别名称到索引的映射，类别为cla，索引为idx

        self.images_path = []  # 存储训练集的所有图片路径
        self.images_label = []  # 存储训练集图片对应索引信息
        self.images_num = []  # 存储每个类别的样本总数
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
        # 遍历每个文件夹下的文件
        for cla in flower_class:
            cla_path = os.path.join(data_path, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(data_path, cla, i)
                      for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            # splitext()函数返回文件名和扩展名组成的元组，[-1]表示取最后一个元素即扩展名；这里是在判断文件是否为支持的文件格式类型；如果是就添加这个路径data_path/cla/i到images列表中；最后的images就是该类别下所有图片的路径列表例如：['data/train/0/1.jpg', 'data/train/1/2.jpg', ...]
            image_class = class_indices[cla]
            # 返回image_class即这个cla类别对应的索引，例如0，1，2，3，4

            # 记录该类别的样本数量
            self.images_num.append(len(images))
            # 写入列表
            for img_path in images:
                self.images_path.append(img_path)
                self.images_label.append(image_class)

        print("{} images were found in the dataset.".format(sum(self.images_num)))  # 打印最后一共有找到多少个dataset在文件夹下

    def __len__(self):
        return sum(self.images_num)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        label = self.images_label[idx]
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
        if self.transform is not None:
            img = self.transform(img)
        else:
            raise ValueError('Image is not preprocessed')
        return img, label

    # 以下是su新增代码，仅用于testing的时候使用原始图片，不进行transform预处理
    def __getitemorig__(self, idx):
        img = Image.open(self.images_path[idx])
        label = self.images_label[idx]
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
        return img, label

    # 非必须实现，torch里有默认实现；该函数的作用是: 决定一个batch的数据以什么形式来返回数据和标签
    # 官方实现的default_collate可以参考
    # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    # 当多个img组合成一个batch的时候，loader会自动调用这个函数，来实现组成batch的操作；注意和前面的__getitem__区分开，getitem是dataset的操作，但是loader会取batch，然后按照batch大小进行给数据进而按照batch进行训练;
    # 通常batch会是这样的格式：[dataset[m],dataset[n],dataset[p],...]，这是loader中是这样取到的，进一步的这里再把他还原给list images和labels用来给机器读取和学习；


# 以下是新增的定义类别用的代码
target_name_soc = ('2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234')
target_name_eoc_1 = ('2S1', 'BRDM2', 'T72', 'ZSU234')

target_name_eoc_2 = ('BMP2', 'BRDM2', 'BTR70', 'T72')
target_name_eoc_2_cv = ('T72-A32', 'T72-A62', 'T72-A63', 'T72-A64', 'T72-S7')
target_name_eoc_2_vv = ('BMP2-9566', 'BMP2-C21', 'T72-812', 'T72-A04', 'T72-A05', 'T72-A07', 'T72-A10')

target_name_confuser_rejection = ('BMP2', 'BTR70', 'T72', '2S1', 'ZIL131')

target_name = {
    'soc': target_name_soc,
    'eoc-1': target_name_eoc_1,
    'eoc-1-t72-132': target_name_eoc_1,
    'eoc-1-t72-a64': target_name_eoc_1,
    'eoc-2-cv': target_name_eoc_2 + target_name_eoc_2_cv,
    'eoc-2-vv': target_name_eoc_2 + target_name_eoc_2_vv,
    'confuser-rejection': target_name_confuser_rejection
}

serial_number = {
    'b01': 0,

    '9563': 1,
    '9566': 1,
    'c21': 1,

    'E-71': 2,
    'k10yt7532': 3,
    'c71': 4,
    '92v13015': 5,
    'A51': 6,

    '132': 7,
    '812': 7,
    's7': 7,
    'A04': 7,
    'A05': 7,
    'A07': 7,
    'A10': 7,
    'A32': 7,
    'A62': 7,
    'A63': 7,
    'A64': 7,

    'E12': 8,
    'd08': 9
}
