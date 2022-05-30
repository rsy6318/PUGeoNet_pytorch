import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class pugeonet(nn.Module):
    def __init__(self, up_ratio, knn=30, fd=64, fD=1024):
        super(pugeonet, self).__init__()
        self.knn = knn
        self.up_ratio = up_ratio
        self.feat_list = ["expand", "net_max_1", "net_mean_1",
                          "out3", "net_max_2", "net_mean_2",
                          "out5", "net_max_3", "net_mean_3",
                          "out7", "out8"]
        #self.input_transform_net = input_transform_net(6)

        self.dgcnn_conv1 = nn.Sequential(nn.Conv2d(6, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv2 = nn.Sequential(nn.Conv2d(fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv3 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv4 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv5 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv6 = nn.Sequential(nn.Conv2d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm2d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv7 = nn.Sequential(nn.Conv1d(fd + fd, fd, kernel_size=1),
                                         nn.BatchNorm1d(fd),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dgcnn_conv8 = nn.Sequential(nn.Conv1d(fd + fd + fd, fD, kernel_size=1),
                                         nn.BatchNorm1d(fD),
                                         nn.LeakyReLU(negative_slope=0.2))

        self.attention_conv1 = nn.Sequential(nn.Conv1d(fd * 9 + fD * 2, 128, kernel_size=1),
                                             nn.BatchNorm1d(128),
                                             nn.LeakyReLU(negative_slope=0.2))
        self.attention_conv2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1),
                                             nn.BatchNorm1d(64),
                                             nn.LeakyReLU(negative_slope=0.2))
        self.attention_conv3 = nn.Sequential(nn.Conv1d(64, len(self.feat_list), kernel_size=1),
                                             nn.BatchNorm1d(len(self.feat_list)),
                                             nn.LeakyReLU(negative_slope=0.2))

        self.concat_conv = nn.Sequential(nn.Conv1d(fd * 9 + fD * 2, 256, kernel_size=1),
                                         nn.BatchNorm1d(256),
                                         nn.LeakyReLU(negative_slope=0.2))
        self.dg1 = nn.Sequential()

        self.uv_conv1 = nn.Sequential(nn.Conv1d(256, up_ratio * 2, kernel_size=1))

        self.patch_conv1 = nn.Sequential(nn.Conv1d(256, 9, kernel_size=1))

        self.normal_offset_conv1 = nn.Sequential(nn.Conv1d(256, up_ratio * 3, kernel_size=1))

        self.up_layer1 = nn.Sequential(nn.Conv2d(256 + 3, 128, kernel_size=1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.up_dg1 = nn.Sequential()
        self.up_layer2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.up_dg2 = nn.Sequential()

        self.fc_layer = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # x:(B,3,N)
        batch_size = x.size(0)
        num_point = x.size(2)
        edge_feature = get_graph_feature(x, k=self.knn)
        '''transform = self.input_transform_net(edge_feature)

        point_cloud_transformed = torch.bmm(x.transpose(1, 2), transform)
        point_cloud_transformed = point_cloud_transformed.transpose(1, 2)
        edge_feature = get_graph_feature(point_cloud_transformed, k=self.knn)'''
        out1 = self.dgcnn_conv1(edge_feature)
        out2 = self.dgcnn_conv2(out1)
        net_max_1 = out2.max(dim=-1, keepdim=False)[0]
        net_mean_1 = out2.mean(dim=-1, keepdim=False)

        out3 = self.dgcnn_conv3(torch.cat((net_max_1, net_mean_1), 1))

        edge_feature = get_graph_feature(out3, k=self.knn)
        out4 = self.dgcnn_conv4(edge_feature)

        net_max_2 = out4.max(dim=-1, keepdim=False)[0]
        net_mean_2 = out4.mean(dim=-1, keepdim=False)

        out5 = self.dgcnn_conv5(torch.cat((net_max_2, net_mean_2), 1))

        edge_feature = get_graph_feature(out5)
        out6 = self.dgcnn_conv6(edge_feature)

        net_max_3 = out6.max(dim=-1, keepdim=False)[0]
        net_mean_3 = out6.mean(dim=-1, keepdim=False)

        out7 = self.dgcnn_conv7(torch.cat((net_max_3, net_mean_3), dim=1))

        out8 = self.dgcnn_conv8(torch.cat((out3, out5, out7), 1))

        out_max = out8.max(dim=-1, keepdim=True)[0]  # B,C

        expand = out_max.repeat(1, 1, num_point)

        concat_unweight = torch.cat((expand,  # 1024
                                     net_max_1,  # 64
                                     net_mean_1,
                                     out3,  # 64
                                     net_max_2,
                                     net_mean_2,
                                     out5,  # 64
                                     net_max_3,
                                     net_mean_3,
                                     out7,  # 64
                                     out8  # 1024
                                     ), dim=1)  # (B,C,N)
        out_attention = self.attention_conv1(concat_unweight)
        out_attention = self.attention_conv2(out_attention)
        out_attention = self.attention_conv3(out_attention)  # (B,C,N)
        out_attention = out_attention.max(dim=-1, keepdim=False)[0]  # (B,C)
        out_attention = F.softmax(out_attention, dim=-1)  # (B,C)

        for i in range(len(self.feat_list)):
            tmp1 = out_attention[:, i]
            dim = eval('%s.size(1)' % self.feat_list[i])
            tmp2 = tmp1.unsqueeze(-1).repeat(1, dim)
            if i == 0:
                attention_weight = tmp2
            else:
                attention_weight = torch.cat((attention_weight, tmp2), axis=-1)
        attention_weight = attention_weight.unsqueeze(-1)
        concat = attention_weight * concat_unweight  # (B,C,N)
        concat = self.concat_conv(concat)
        concat = self.dg1(concat)  # (B,C,N)

        uv_2d = self.uv_conv1(concat)
        uv_2d = uv_2d.reshape(batch_size, self.up_ratio, 2, num_point)  # B,U,2,N
        uv_2d = torch.cat((uv_2d, torch.zeros((batch_size, self.up_ratio, 1, num_point)).to(x.device)),
                          dim=2)  # B,U,3,N

        affine_T = self.patch_conv1(concat)
        affine_T = affine_T.reshape(batch_size, 3, 3, num_point)  # B,3,3,N

        uv_3d = torch.matmul(uv_2d.permute(0, 3, 1, 2), affine_T.permute(0, 3, 1, 2))  # B, N, U, 3
        uv_3d = uv_3d.permute(0, 2, 3, 1)  # (B,U,3,N)
        uv_3d = x.unsqueeze(1).repeat(1, self.up_ratio, 1, 1) + uv_3d  # (B,U,3,N)

        uv_3d = uv_3d.transpose(1,2)        #(B,3,U,N)

        # B,3,U,N
        #uv_3d = uv_3d.permute(0, 2, 1, 3).reshape(batch_size, 3, self.up_ratio * num_point)

        # norm predict
        dense_normal_offset = self.normal_offset_conv1(concat)
        dense_normal_offset = dense_normal_offset.reshape(batch_size, self.up_ratio, 3, num_point)

        sparse_normal = torch.from_numpy(np.array([0, 0, 1]).astype(np.float32)).squeeze().reshape(1, 1, 3, 1).repeat(
            batch_size, 1, 1, num_point).to(x.device)

        sparse_normal = torch.matmul(sparse_normal.permute(0, 3, 1, 2), affine_T.permute(0, 3, 1, 2))
        sparse_normal = sparse_normal.permute(0, 2, 3, 1)
        sparse_normal = F.normalize(sparse_normal, dim=2)  # B,1,3,N

        dense_normal = sparse_normal.repeat(1, self.up_ratio, 1, 1, ) + dense_normal_offset
        dense_normal = F.normalize(dense_normal, dim=2)  # B, U, 3, N

        dense_normal=dense_normal.transpose(1,2).reshape(batch_size,3,-1)        #(B,U,3,N)->(B,3,U,N)->(B,3,U*N)

        grid = uv_3d        #(B,3,U,N)

        #concat_up = concat.repeat(1, 1, self.up_ratio)
        concat_up=concat.unsqueeze(2).repeat(1,1,self.up_ratio,1)   #(B,C,U,N)
        concat_up = torch.cat((concat_up, grid), axis=1)            #(B,C+3,U,N)

        concat_up = self.up_layer1(concat_up)
        concat_up = self.up_dg1(concat_up)
        concat_up = self.up_layer2(concat_up)
        concat_up = self.up_dg2(concat_up)

        coord_z = self.fc_layer(concat_up)  #(B,1,U,N)


        coord_z = torch.cat((torch.zeros_like(coord_z), torch.zeros_like(coord_z), coord_z), dim=1)  # B,3,U,N

        coord_z = torch.matmul(coord_z.permute(0, 3, 2, 1), affine_T.permute(0, 3, 1, 2))  # B,N,U,3
        
        coord = uv_3d + coord_z.permute(0, 3, 2, 1)     #(B,3,U,N)

        coord=coord.reshape(batch_size,3,-1)

        return {'dense_xyz':coord, 'dense_normal':dense_normal, 'sparse_normal':sparse_normal.squeeze(1)}

