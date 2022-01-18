import torch
import torch.nn as nn
from einops import rearrange

from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils.PyTorchEMD.emd import earth_mover_distance


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()

    def forward(self,target,pre):
        # TODO：loss函数要改写，要计算每个点的cd、emd距离 （已完成）
        # cd距离
        B = target.shape[0]
        # print("pre[0]",pre[0].shape)
        # print("target[0]",target[0].shape)
        cd_total_loss = 0
        for i in range(B):
            chamloss = dist_chamfer_3D.chamfer_3DDist()
            pre1 = rearrange(pre[i],'(b c) h -> b c h', b=1)
            target1 = rearrange(target[i],'(b c) h -> b c h', b=1)
            dist1,dist2,idx1,idx2 = chamloss(pre1,target1) # 这个地方的错误是有3个维度，pre[i]之后就是2个维度了
            # print("dist1",dist1.shape) # 原来的[8,4800]
            cd_loss = torch.mean(dist1)+0.5*torch.mean(dist2)
            cd_total_loss += cd_loss

        # emd距离
        d = earth_mover_distance(pre,target,transpose=False)

        emd_loss = torch.sum(d)
        # print(emd_loss.shape)
        # emd_loss = d[0] / 2 + d[1] * 2 + d[2] / 3
        # print("emd_loss", cd_loss)

        total_loss = cd_total_loss + emd_loss
        return total_loss
