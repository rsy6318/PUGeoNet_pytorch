import os
import torch
#import dataloader as provider
#import tensorflow as tf
import numpy as np
import argparse
import model
import loss
import logging
from glob import glob
from pc_util import  normalize_point_cloud, farthest_point_sample, group_points
import open3d as o3d
#print(torch.cuda.is_available())


def eval_patches(xyz, arg, model):
    centroids = farthest_point_sample(xyz, arg.num_patch)

    '''pointcloud1=o3d.geometry.PointCloud()
    pointcloud1.points=o3d.utility.Vector3dVector(xyz)
    pointcloud1.paint_uniform_color([0,0,1])

    pointcloud2 = o3d.geometry.PointCloud()
    pointcloud2.points = o3d.utility.Vector3dVector(centroids)
    pointcloud2.paint_uniform_color([0, 1, 1])

    o3d.visualization.draw_geometries([pointcloud2,pointcloud1])'''

    patches = group_points(xyz, centroids, arg.num_point)
    '''print(patches.shape)
    pointcloud1 = o3d.geometry.PointCloud()
    pointcloud1.points = o3d.utility.Vector3dVector(patches[0])
    pointcloud1.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([ pointcloud1])'''

    normalized_patches, patch_centroid, furthest_distance = normalize_point_cloud(patches)

    dense_patches_list = []
    dense_normal_list = []
    sparse_normal_list = []

    #print(normalized_patches.shape)

    for i in range(normalized_patches.shape[0]):
        xyz=torch.from_numpy(normalized_patches[i]).unsqueeze(0).transpose(1, 2).cuda()
        #print(torch.from_numpy(normalized_patches[i]).size())
        dense_patches, dense_normal, sparse_normal = model(xyz, 1)
        dense_patches_list.append(dense_patches.transpose(1, 2).detach().cpu().numpy())
        dense_normal_list.append(dense_normal.transpose(1, 2).detach().cpu().numpy())
        sparse_normal_list.append(sparse_normal.transpose(1, 2).detach().cpu().numpy())

    gen_ddense_xyz = np.concatenate(dense_patches_list, axis=0)
    gen_ddense_xyz = (gen_ddense_xyz * furthest_distance) + patch_centroid
    gen_ddense_normal = np.concatenate(dense_normal_list, axis=0)

    return np.reshape(gen_ddense_xyz, (-1, 3)), np.reshape(gen_ddense_normal, (-1, 3))


def evaluate(model, arg):
    model.eval()
    shapes = glob(arg.eval_xyz + '/*.xyz')

    for i, item in enumerate(shapes):
        #print(item)
        obj_name = item.split('/')[-1]
        data = np.loadtxt(item)
        input_sparse_xyz = data[:, 0:3]
        input_sparse_normal = data[:, 3:6]
        normalize_sparse_xyz, centroid, furthest_distance = normalize_point_cloud(input_sparse_xyz)
        dense_xyz, dense_normal = eval_patches(normalize_sparse_xyz, arg,model)
        dense_xyz = dense_xyz * furthest_distance + centroid
        gen_dense=np.concatenate((dense_xyz,dense_normal),axis=-1)
        #print(gen_dense.shape)
        #print(arg.eval_save_path)
        savepath=os.path.join(arg.eval_save_path,obj_name)
        #print(arg.eval_save_path)
        #print(savepath)
        
        #print(gen_dense.shape)
        gen_dense=farthest_point_sample(gen_dense,arg.num_shape_point*arg.up_ratio)
        #print(gen_dense.shape)
        np.savetxt(savepath,gen_dense)
        print(obj_name,'is saved')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--up_ratio', type=int, default=4,choices=[4,8,12,16], help='Upsampling Ratio')  
    parser.add_argument('--model', default='model_pugeo', help='Model for upsampling')
    parser.add_argument('--num_point', type=int, default=256,choices=[256], help='Point Number')
    
    parser.add_argument('--eval_xyz', default='/home/siyu_ren/pugeo_pytorch_data/test_5000/', help='Folder to store point cloud(xyz format) toevaluate')
    parser.add_argument('--num_shape_point', type=int, default=5000, help='Point Number per shape')
    parser.add_argument('--patch_num_ratio', type=int, default=3, help='Number of points covered by patch')
    arg = parser.parse_args()
    arg.log_dir='log_x%d'%arg.up_ratio
    arg.num_patch = int(arg.num_shape_point / arg.num_point * arg.patch_num_ratio)
    arg.eval_save_path='PUGEOx%d'%arg.up_ratio
    try:
        os.mkdir(arg.eval_path)
    except:
        pass
    model = model.pugeonet(up_ratio=arg.up_ratio, knn=30)
    model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(arg.log_dir,'model_500.t7')))
    evaluate(model, arg)
