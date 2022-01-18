import torch
import torch.nn as nn
from einops import rearrange
from torchvision.transforms import Compose,ToTensor,Resize
from PIL import Image
import numpy as np
from dataloader.show3d_balls import showpoints
model_path = r'C:\Anaconda\envs\torch1.9\pyproject\paper_project\MyProject\save_model\model.pth'
Model = torch.load(model_path)
Model.cuda()

img_path = r"D:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0\02691156\seg_img\1af4b32eafffb0f7ee60c37cbf99c1c.png"
transfer = Compose([ToTensor(),
                    Resize([224,224])])
img = Image.open(img_path)
img = transfer(img).cuda()

print("img_shape",img.shape)

img = rearrange(img,'(b c) h w ->b c h w ',b=1,c=3)
print("img",img.shape)
# test_data = torch.randn(16,3,224,224)
# test_data = test_data.cuda()
output = Model(img)
print(output.shape)

point_show = rearrange(output, 'b n m -> (b n ) m')
# seg = rearrange(img,'b n -> ( b n)') # 在原show3d_balls中,数据获取没有用dataloader,没有batch_size所以要转换格式回去

point_show = point_show.cpu().detach().numpy() #[2500,3]
# print(point_show.shape)
# seg = seg.numpy() #[2500,]
# print(seg.shape)
choice = np.random.choice(point_show.shape[0], 4800, replace=True)  # 采样的点
# print("choice",choice)
point_set = point_show[choice, :]

showpoints(point_set, None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
           background=(255, 255, 255), normalizecolor=True, ballradius=10)
