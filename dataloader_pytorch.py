import numpy as np
import os
import torch
import torch.utils.data as data
import open3d as o3d

class dataset_patch(data.Dataset):
    def __init__(self, up_ratio, data_path,jitter=False,jitter_sigma=0.03,jitter_max=0.05):
        super(dataset_patch,self).__init__()
        assert up_ratio in [4,8,12,16], 'upratio should be one of [4, 8, 12, 16]'
        self.jitter=jitter
        self.jitter_sigma = jitter_sigma
        self.jitter_max = jitter_max
        basic_root=os.path.join(data_path,'basic')
        label_root=os.path.join(data_path,'%d'%up_ratio)
        name_list=os.listdir(basic_root)

        self.basic_set=[]
        self.label_set=[]
        for name in name_list:
            self.basic_set.append(np.load(os.path.join(basic_root,name)))
            self.label_set.append(np.load(os.path.join(label_root,name)))

        self.basic_set=np.concatenate(self.basic_set,axis=0)
        self.label_set=np.concatenate(self.label_set,axis=0)
        #print(self.label_set.shape)
    def __len__(self):
        return self.basic_set.shape[0]

    def rotate_point_cloud_and_gt(self,input,sparse_normal,label,label_normal):
        angles=np.random.uniform(0,1,(3,))*np.pi*2
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        input=np.dot(input,R)
        label=np.dot(label,R)
        sparse_normal = np.dot(sparse_normal, R)
        label_normal= np.dot(label_normal, R)
        return input,sparse_normal,label,label_normal

    def random_scale_point_cloud_and_gt(self,input,label,scale_low=0.5,scale_high=2.0):
        scale=np.random.uniform(scale_low,scale_high)
        input=input*scale
        label=label*scale
        return input,label,scale

    def jitter_perturbation_point_cloud(self,input,sigma=0.005,clip=0.02):
        assert clip>0
        jittered_data=np.clip(sigma*np.random.normal(size=input.shape),-1*clip,clip)
        #print(input.shape)
        #print(jittered_data.shape)
        input=input+jittered_data
        return input


    def augment_data(self,input,sparse_normal,label,label_normal):
        input,sparse_normal,label,label_normal=self.rotate_point_cloud_and_gt(input,sparse_normal,label,label_normal)
        input,label,scale=self.random_scale_point_cloud_and_gt(input,label,scale_low=0.8,scale_high=1.2)
        if self.jitter:
            input=self.jitter_perturbation_point_cloud(input,sigma=self.jitter_sigma,clip=self.jitter_max)
        return input,sparse_normal,label,label_normal,scale


    def __getitem__(self, item):
        # return input sparse patch, gt sparse normal, gt dense patch, gt dense normal
        input_sparse_patch,gt_sparse_normal,gt_dense_patch,gt_dense_normal=self.basic_set[item,:,0:3],self.basic_set[item,:,3:],self.label_set[item,:,0:3],self.label_set[item,:,3:]

        input_sparse_patch, gt_sparse_normal, gt_dense_patch, gt_dense_normal,radius=self.augment_data(input_sparse_patch,gt_sparse_normal,gt_dense_patch,gt_dense_normal)

        return torch.from_numpy(input_sparse_patch.astype(np.float32)).transpose(0,1),torch.from_numpy(gt_sparse_normal.astype(np.float32)).transpose(0,1),torch.from_numpy(gt_dense_patch.astype(np.float32)).transpose(0,1),torch.from_numpy(gt_dense_normal.astype(np.float32)).transpose(0,1),torch.from_numpy(np.array(radius).astype(np.float32))

if __name__=='__main__':
    dataset=dataset_patch(16,'D:\\PUGEO\\pytorch_data',jitter=True)
    for i in range(len(dataset)):
        data=dataset[i]

        input=data[0]
        '''sparse_normal=data[1]
        dense_patch=data[2]
        dense_normal=data[3]
    
        data = dataset[1]
    
        input2 = data[0]
        sparse_normal2 = data[1]
        dense_patch2 = data[2]
        dense_normal2 = data[3]'''

        print(i,' ',input.size())
    #print(dense_patch.size())

    '''point_cloud1=o3d.geometry.PointCloud()
    point_cloud1.points=o3d.utility.Vector3dVector(input)
    point_cloud1.paint_uniform_color([0,0,1])

    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(dense_patch)
    point_cloud2.paint_uniform_color([0, 1, 1])

    o3d.visualization.draw_geometries([point_cloud1,point_cloud2])'''