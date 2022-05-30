import numpy as np
import open3d as o3d
import os
import argparse
from scipy.stats import entropy
import warnings
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import xlsxwriter
import pymesh


def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance


def cal_cd_hd(result,gt):
    #N=result.shape[0]
    dist=np.sqrt(np.sum((np.expand_dims(result,0)-np.expand_dims(gt,1))**2,axis=-1))
    dist1=np.min(dist,axis=0,keepdims=False)
    dist2=np.min(dist,axis=1,keepdims=False)
    cd=np.mean(dist1)+np.mean(dist2)
    hd=0.5*(np.amax(dist1,axis=0)+np.amax(dist2,axis=0))
    return cd,hd

def cal_cd_hd2(result,gt):
    #N=result.shape[0]
    pc_result=o3d.geometry.PointCloud()
    pc_gt=o3d.geometry.PointCloud()
    pc_result.points=o3d.utility.Vector3dVector(result)
    pc_gt.points=o3d.utility.Vector3dVector(gt)

    tree_gt=o3d.geometry.KDTreeFlann(pc_gt)
    dist1=[]
    for i in range(result.shape[0]):
        [k,idx,dist]=tree_gt.search_knn_vector_3d(result[i],1)
        dist1.append(np.sqrt(np.array(dist)))

    dist1=np.array(dist1)

    tree_result = o3d.geometry.KDTreeFlann(pc_result)
    dist2 = []
    for i in range(gt.shape[0]):
        [k, idx, dist] = tree_result.search_knn_vector_3d(gt[i], 1)
        dist2.append(np.sqrt(np.array(dist)))

    dist2 = np.array(dist2)

    '''assert False
    dist=np.sqrt(np.sum((np.expand_dims(result,0)-np.expand_dims(gt,1))**2,axis=-1))
    dist1=np.min(dist,axis=0,keepdims=False)
    dist2=np.min(dist,axis=1,keepdims=False)'''
    cd=np.mean(dist1)+np.mean(dist2)
    hd=0.5*(np.amax(dist1,axis=0)+np.amax(dist2,axis=0))
    return cd,hd

def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=True):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters

def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _jsdiv(P, Q):
    '''another way of computing JSD'''
    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))

def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)      # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res

def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.    
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    '''   
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/siyuren_21/pytorch_data/', help='data path')
    parser.add_argument('--up_ratio', type=int,  choices=[4,8,12,16],default=4, help='data path')
    parser.add_argument('--num_shape_point',choices=[5000], type=int, default=5000, help='data path')

    arg = parser.parse_args()

    result_path=os.path.join(os.path.dirname(__file__),'..','PUGEOx%d/'%arg.up_ratio)
    gt_path=os.path.join(arg.data_path,'test_%d'%int(arg.up_ratio*arg.num_shape_point))
    #'/home/siyu_ren/pugeo_pytorch_data/test_20000/'

    name_list=os.listdir(gt_path)

    wb = xlsxwriter.Workbook('./result_%d.xls'%arg.up_ratio)

    workbook=wb.add_worksheet('result')

    ratio_raw_dict={'4':1,'8':2,'12':3,'16':4}

    cd_result=[]
    hd_result=[]
    p2f_result=[]
    jsd_result=[]
    for name in name_list:
        result=np.loadtxt(os.path.join(result_path,name))[:,0:3]
        gt=np.loadtxt(os.path.join(gt_path,name))[:,0:3]
        cd,hd=cal_cd_hd2(result,gt)
        cd_result.append(cd)
        hd_result.append(hd)

        jsd=jsd_between_point_cloud_sets(np.expand_dims(result,0), np.expand_dims(gt,0))

        #p2f=np.loadtxt(os.path.join(result_path,name[:-4]+'_point2mesh_distance.xyz'))[:,-1]
        p2f,_,_=pymesh.distance_to_mesh(pymesh.meshio.load_mesh(os.path.join(arg.data_path,'test_mesh',name[:-4]+'.off'),drop_zero_dim=False),result,engine='auto')
        p2f=np.sqrt(p2f).squeeze()

        p2f_result.append(p2f)
        jsd_result.append(jsd)
        print(name,' cd:',cd,' hd:',hd, ' jsd:',jsd, ' p2f:',np.mean(p2f),'+-',np.std(p2f))
    cd_result=np.array(cd_result)
    hd_result=np.array(hd_result)
    p2f_result=np.concatenate(p2f_result,axis=0)
    jsd_result=np.array(jsd_result)

    workbook.write(0,1,'cd(1e-2)')
    workbook.write(0,2,'hd(1e-2)')
    workbook.write(0,3,'jsd(1e-2)')
    workbook.write(0,4,'p2f avg(1e-3)')
    workbook.write(0,5,'p2f std(1e-3)')

    workbook.write(ratio_raw_dict['%d'%arg.up_ratio],0,'x%d'%arg.up_ratio)
    workbook.write(ratio_raw_dict['%d'%arg.up_ratio],1,'%0.3f'%(np.mean(cd_result)*1e2))
    workbook.write(ratio_raw_dict['%d'%arg.up_ratio],2,'%0.3f'%(np.mean(hd_result)*1e2))
    workbook.write(ratio_raw_dict['%d'%arg.up_ratio],3,'%0.3f'%(np.mean(jsd_result)*1e2))
    workbook.write(ratio_raw_dict['%d'%arg.up_ratio],4,'%0.3f'%(np.nanmean(p2f_result)*1e3))
    workbook.write(ratio_raw_dict['%d'%arg.up_ratio],5,'%0.3f'%(np.nanstd(p2f_result)*1e3))


    wb.close()
    print('cd',round(np.mean(cd_result),7))
    print('hd',round(np.mean(hd_result),7))
    print('jsd',round(np.mean(jsd_result),7))
    print('p2f avg',round(np.nanmean(p2f_result),7))
    print('p2f std',round(np.nanstd(p2f_result),7))
