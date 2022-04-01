import torch
import os
import numpy as np
from sklearn.neighbors import KDTree

def nn_distance(xyz1,xyz2):
    #xyz1(B,3,N1)
    #xyz2(B,3,N2)
    square_dist=torch.sum((xyz1.unsqueeze(-1)-xyz2.unsqueeze(-2))**2,dim=1,keepdim=False)
    dist1,idx1=square_dist.min(dim=-1,keepdim=False)
    dist2,idx2=square_dist.min(dim=-2,keepdim=False)

    return dist1,idx1,dist2,idx2

def cd_loss(gen,gt,radius,ration=0.5):
    dists_forward,idx1,dists_backward,idx2=nn_distance(gt,gen)
    cd_dist = 0.5*dists_forward + 0.5*dists_backward
    cd_dist = torch.mean(cd_dist, dim=1)

    cd_dist_norm = cd_dist/radius
    cd_loss=torch.mean(cd_dist_norm)

    return cd_loss,idx1,idx2

def abs_dense_normal_loss(gen_normal, gt_normal, idx1, idx2, radius, ratio=0.5):
    #gen_normal B,3,N
    fwd1=torch.gather(gen_normal,dim=2,index=idx1.unsqueeze(1).repeat(1,3,1))
    pos_dist1 = torch.mean((gt_normal - fwd1) ** 2,dim=1)
    neg_dist1 = torch.mean((gt_normal + fwd1) ** 2,dim=1)

    dist1=torch.where(pos_dist1<neg_dist1,pos_dist1,neg_dist1)
    dist1=torch.mean(dist1,dim=1)

    fwd2=torch.gather(gt_normal,dim=2,index=idx2.unsqueeze(1).repeat(1,3,1))
    pos_dist2 = torch.mean((gen_normal - fwd2) ** 2,dim=1)
    neg_dist2 = torch.mean((gen_normal + fwd2) ** 2,dim=1)

    dist2 = torch.where(pos_dist2 < neg_dist2, pos_dist2, neg_dist2)
    dist2 = torch.mean(dist2,dim=1)

    dist=0.5*dist1+0.5*dist2

    dist_norm=dist/radius

    normal_loss=torch.mean(dist_norm)
    return normal_loss

def abs_sparse_normal_loss(gen_normal,gt_normal,radius):
    pos_dist=torch.mean((gt_normal-gen_normal)**2,dim=1)
    neg_dist=torch.mean((gt_normal+gen_normal)**2,dim=1)
    dist = torch.where(pos_dist < neg_dist, pos_dist, neg_dist)

    dist = torch.mean(dist,dim=-1)

    dist_norm=dist/radius

    return torch.mean(dist_norm)