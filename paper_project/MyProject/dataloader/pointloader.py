# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
import cv2
from einops import rearrange
from torch.utils.data import Dataset
from .show3d_balls import showpoints
from torchvision.transforms import ToPILImage,Resize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = 'D:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0', npoints=1024, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints # 采样点数
        self.root = root # 文件根路径
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt') # 类别和文件夹名字对应的路径
        self.cat = {
    }
        self.normal_channel = normal_channel # 是否使用rgb信息


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {
    k: v for k, v in self.cat.items()} #{'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
        self.classes_original = dict(zip(self.cat, range(len(self.cat)))) #{'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        # print("self.cat",self.cat)
        # print("self.classes_original",self.classes_original)
        if not class_choice is  None:  # 选择一些类别进行训练  好像没有使用这个功能
            self.cat = {
    k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {
    } # 读取分好类的文件夹jason文件 并将他们的名字放入列表中

        self.meta2 = {
        }
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)]) # '928c86eabc0be624c2bf2dcc31ba1713' 这是第一个值
            # print("train_ids",train_ids)
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = [] # 字典,存放形如{'Airplane': ['D:\\Dataset\\shapenetcore_partanno_segmentation_benchmark_v0\\shapenetcore_partanno_segmentation_benchmark_v0\\02691156\\points\\1021a0914a7207aff927ed529ad90a11.pts',....
            self.meta2[item] = [] #存放图片路径
            dir_point = os.path.join(self.root, self.cat[item]) # # 拿到对应一个文件夹的路径 例如第一个文件夹02691156
            # print("dir_point",dir_point)
            classes = sorted(os.listdir(dir_point)) # 数据集集三个类别points', 'points_label', 'seg_img
            # 先默认选择point
            point_path = os.path.join(dir_point,classes[0])
            # print(point_path)
            # 选择图片文件夹
            image_path = os.path.join(dir_point,classes[2])

            fns = sorted(os.listdir(point_path))  # 根据路径拿到文件夹下的每个pts文件 放入列表中
            ins = sorted(os.listdir(image_path))  # 根据路径拿到文件夹下的每个png文件 放入列表中
            # print(ins)
            # print(ins[0][0:-4])
            # print("fns",fns)
            # print(fns[0][0:-4])

            # points操作
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
                # ins = [in_ for in_ in ins if ((in_[0:-4] in train_ids) or ((in_[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids] # 判断文件夹中的txt文件是否在 训练txt中，如果是，那么fns中拿到的txt文件就是这个类别中所有txt文件中需要训练的文件，放入fns中
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                "第i次循环  fns中拿到的是第i个文件夹中符合训练的txt文件夹的名字"
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(point_path, token + '.pts'))  # 生成一个字典，将类别名字和训练的路径组合起来  作为一个大类中符合训练的数据
                #上面的代码执行完之后，就实现了将所有需要训练或验证的数据放入了一个字典中，字典的键是该数据所属的类别，例如飞机。值是他对应数据的全部路径
                #{Airplane:[路径1，路径2........]}
                # print("meta",self.meta)
                self.meta2[item].append(os.path.join(image_path, token + '.png')) # 生成图片字典
                # print("meta2",self.meta2[item])
        #####################################################################################################################################################
        self.datapath = []
        for item in self.cat: # self.cat 是类别名称和文件夹对应的字典
            for fn in self.meta[item]:
                self.datapath.append((item, fn)) # 生成标签和点云路径的元组， 将self.met 中的字典转换成了一个元组
                # print(self.datapath)

        #####################################################################################################################################################
        self.imgpath = []
        for item in self.cat:  # self.cat 是类别名称和文件夹对应的字典
            for fn in self.meta2[item]:
                self.imgpath.append((item, fn))  # 生成标签和图片路径的元组， 将self.met 中的字典转换成了一个元组
                # print(self.imgpath)


        self.classes = {
    }
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        ## self.classes  将类别的名称和索引对应起来  例如 飞机 <----> 0
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        """
        shapenet 有16 个大类，然后每个大类有一些部件 ，例如飞机 'Airplane': [0, 1, 2, 3] 其中标签为0 1  2 3 的四个小类都属于飞机这个大类
        self.seg_classes 就是将大类和小类对应起来
        """
        self.seg_classes = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {
    }  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000




    def __getitem__(self, index):
        if index in self.cache: # 初始slef.cache为一个空字典，这个的作用是用来存放取到的数据，并按照(point_set, cls, seg,img_data)放好 同时避免重复采样
            point_set, cls, seg, img_data = self.cache[index]
        else:
            fn = self.datapath[index] # 根据索引 拿到训练数据的路径self.datepath是一个元组（类名，路径）
            im = self.imgpath[index] # 根据索引 拿到训练数据的路径self.imgpath是一个元组（类名，路径）

            # print(fn) # ('Airplane', 'D:\\Dataset\\shapenetcore_partanno_segmentation_benchmark_v0\\shapenetcore_partanno_segmentation_benchmark_v0\\02691156\\points\\26210ec84a9c1c6eb1bb46d2556ba67d.pts')
            cat = self.datapath[index][0] # 拿到类名
            cls = self.classes[cat] # 将类名转换为索引
            # print(cls)
            cls = np.array([cls]).astype(np.int32)
            # print(fn[1])
            # print(im[1])
            data = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 读入这个txt文件，共20488个点，每个点xyz rgb +小类别的标签
            img_data = cv2.imread(im[1])
            img_data = cv2.resize(img_data,dsize=(224,224))
            img_data = img_data.transpose(2, 0, 1)


            if not self.normal_channel:  # 判断是否使用rgb信息
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32) # 拿到小类别的标签
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, img_data)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # 做一个归一化

        choice = np.random.choice(len(seg), self.npoints, replace=True) # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        # resample
        point_set = point_set[choice, :] # 根据索引采样
        seg = seg[choice]

        return point_set, cls, seg , img_data# pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return len(self.datapath)

if __name__ == '__main__':
    import torch

    root = r'D:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0'
    "测试一下sharpnet数据集"
    data = PartNormalDataset(root=root)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=False)
    for point in DataLoader:
        print('point0.shape:\n', point[0].shape) # ([2, 2500, 3])
        print('point1.shape:\n', point[1].shape) # [2, 1])  大部件的标签
        print('point2.shape:\n', point[2].shape)  # torch.Size([2, 2500])  部件类别标签，每个点的标签
        print("image:\n",point[3].shape)
        #print('label.shape:\n', label.shape)

        point_show = point[0]
        point_img = point[3]
        break
    # 用show3d_ball方法
    point_show = rearrange(point_show, 'b n m -> (b n ) m')
    seg = rearrange(point[2],'b n -> ( b n)') # 在原show3d_balls中,数据获取没有用dataloader,没有batch_size所以要转换格式回去
    point_show = point_show.numpy() #[2500,3]
    # print(point_show.shape)
    seg = seg.numpy() #[2500,]
    # print(seg.shape)
    cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                     [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                     [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                     [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])
    choice = np.random.choice(point_show.shape[0], 2500, replace=True)  # 采样的点
    # print("choice",choice)
    point_set, seg = point_show[choice, :], seg[choice]
    seg = seg - seg.min()
    gt = cmap[seg, :]
    # print("gt",gt)
    pred = cmap[seg, :]
    # print("pred",pred)
    showpoints(point_set, None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(255, 255, 255), normalizecolor=True, ballradius=10)
    point_img = rearrange(point_img,'b c h w -> (b c) h w')
    show = ToPILImage()
    show(point_img).show()


    # point_show = rearrange(point_show,'b n m -> (b n ) m')
    # 用matplotlib的方式
    # # print("point_show",point_show.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # x = []
    # y = []
    # z = []
    # for i in range(len(point_show)):
    #     x.append(point_show[i][0])
    #     y.append(point_show[i][1])
    #     z.append(point_show[i][2])
    #
    # ax.scatter(x, y, z, c='k', marker='.', s=0.1)
    #
    # plt.show()

