import os
import open3d as o3d
import pymeshlab
import numpy as np


def mesh2pc(num_point,mode):
    #num_point=20000
    #mode='train'
    data_path='D:\\PUGEO\\mesh\\%s_mesh'%mode
    shape_name_list=os.listdir(data_path)
    save_path=os.path.join(data_path,'..','%s_%d'%(mode,num_point))

    try:
        os.mkdir(save_path)
    except:
        pass

    for shape_name in shape_name_list:
        '''ms=pymeshlab.MeshSet()
        ms.load_new_mesh(os.path.join(data_path,shape_name))
        ms.load_filter_script('F:\\ModelNet40_aligned\\PDS_exact_%d.mlx'%num_point)
        ms.apply_filter_script()'''
        ms=o3d.io.read_triangle_mesh(os.path.join(data_path,shape_name))
        point_cloud=ms.sample_points_poisson_disk(num_point,use_triangle_normal=True)
        #o3d.visualization.draw_geometries([point_cloud])
        #print(pc.shape)
        pc=np.array(point_cloud.points).astype(np.float32)
        normal=np.array(point_cloud.normals).astype(np.float32)
        print(shape_name,num_point)
        #print(pc.shape)
        #print(normal.shape)
        np.savetxt(os.path.join(save_path,shape_name[:-4]+'.xyz'),np.concatenate((pc,normal),axis=-1))


if __name__=='__main__':
    num_point_list=[5000,20000,40000,80000,160000]
    mode_list=['train','test']
    #mesh2pc(5000,'test')
    #assert False
    for num_point in num_point_list:
        for mode in mode_list:
            mesh2pc(num_point, mode)