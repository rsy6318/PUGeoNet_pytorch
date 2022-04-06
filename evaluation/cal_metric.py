import numpy as np
import open3d as o3d
import os
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/siyu_ren/pugeo_pytorch_data', help='data path')
    parser.add_argument('--up_ratio', type=int,  choices=[4,8,12,16],default=4, help='data path')
    parser.add_argument('--num_shape_point',choices=[5000], type=int, default=5000, help='data path')

    arg = parser.parse_args()

    result_path=os.path.join(os.path.dirname(__file__),'..','PUGEOx%d/'%arg.up_ratio)
    gt_path=os.path.join(arg.data_path,'test_%d'%int(arg.up_ratio*arg.num_shape_point))
    #'/home/siyu_ren/pugeo_pytorch_data/test_20000/'

    name_list=os.listdir(gt_path)


    def cal_cd_hd(result,gt):
        #N=result.shape[0]
        dist=np.sqrt(np.sum((np.expand_dims(result,0)-np.expand_dims(gt,1))**2,axis=-1))
        dist1=np.min(dist,axis=0,keepdims=False)
        dist2=np.min(dist,axis=1,keepdims=False)
        cd=np.mean(dist1)+np.mean(dist2)
        hd=0.5*(np.amax(dist1,axis=0)+np.amax(dist2,axis=0))
        
        return cd,hd

    cd_result=[]
    hd_result=[]
    p2f_result=[]
    for name in name_list:
        result=np.loadtxt(os.path.join(result_path,name))[:,0:3]
        gt=np.loadtxt(os.path.join(gt_path,name))[:,0:3]
        cd,hd=cal_cd_hd(result,gt)
        cd_result.append(cd)
        hd_result.append(hd)
        p2f=np.loadtxt(os.path.join(result_path,name[:-4]+'_point2mesh_distance.xyz'))[:,-1]
        p2f=np.mean(p2f)
        p2f_result.append(p2f)

        print(name,' ',cd)
    cd_result=np.array(cd_result)
    hd_result=np.array(hd_result)
    p2f_result=np.array(p2f_result)
    print('cd',np.mean(cd_result))
    print('hd',np.mean(hd_result))
    print('p2f avg',np.mean(p2f_result))
    print('p2f std',np.std(p2f_result))
