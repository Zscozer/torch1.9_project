import torch
from torch.utils.data import DataLoader
from Module import model
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor,Resize,Compose
from dataloader.pointloader import PartNormalDataset
from utils.loss import loss
import os
from torch.nn import CrossEntropyLoss
from torch.optim import SGD,Adam
import Module.PSGN_Module.model as PSGN

# root_dir=r'C:\Anaconda\envs\pytorch\Pyproject\Graduation_project'



# datasets = ImageFolder(os.path.join(root_dir,'chairs'),transform=transformers)
root_dir = r"D:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0"
datasets = PartNormalDataset(root_dir)
# if torch.load(r"C:\Anaconda\envs\torch1.9\pyproject\paper_project\MyProject\save_model\Module.pth"):
#     Model = torch.load(r"C:\Anaconda\envs\torch1.9\pyproject\paper_project\MyProject\save_model\Module.pth")
# else:

# Model = model.Model()
# Model.cuda()

# 使用PSGN的方法 注意:PSGN的采样点是1024，我原模型的采样点是4800
Model = PSGN.create_psgn_occu()
Model.cuda()

epoch = 200
loss = loss()
optim = Adam(Model.parameters(),lr=3e-5,weight_decay=5e-6)
# optim = SGD(Model.parameters(),0.01)

for i in range(epoch):
    total_train_step = 0
    print("-------第{}轮训练------".format(i + 1))
    for data in DataLoader(dataset=datasets,batch_size=8,shuffle=True):
        points = data[0].cuda()
        # print("points",points.shape)
        img = data[3].cuda()
        img = img.float()
        # print(img.shape)
        # output = Model(img)
        # print(output.shape)


        output = Model(img)
        # print(points.shape)

        result_loss = loss(points, output)
        # print("result_loss",result_loss)
        optim.zero_grad()
        result_loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step%10 == 0:
            print("训练次数：{} loss:{}".format(total_train_step, result_loss.item()))  # result_loss为tensor类型，.item转化类型

    torch.save(Model,r"C:\Anaconda\envs\torch1.9\pyproject\paper_project\MyProject\save_model\model.pth")
