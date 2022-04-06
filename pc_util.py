import os
import sys

from sklearn.neighbors import NearestNeighbors
import numpy as np


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def group_points(xyz1,xyz2,k):
    #xyz2: keypoints
    #print(xyz1.shape)
    #print(xyz2.shape)
    square_dists=np.sum((np.expand_dims(xyz1,0)-np.expand_dims(xyz2,1))**2,axis=-1,keepdims=False)
    final_result=np.zeros((xyz2.shape[0],k,3),dtype=np.float32)
    idx=np.argsort(square_dists,axis=1)[:,0:k]     #N2,k
    for i in range(idx.shape[0]):
        final_result[i,:,:]=xyz1[idx[i],:]
    return final_result


def group_points_with_normal(xyz1,xyz2,k):
    #xyz2: keypoints
    #print(xyz1.shape)
    #print(xyz2.shape)
    square_dists=np.sum((np.expand_dims(xyz1[:,0:3],0)-np.expand_dims(xyz2[:,0:3],1))**2,axis=-1,keepdims=False)
    final_result=np.zeros((xyz2.shape[0],k,xyz2.shape[-1]),dtype=np.float32)
    idx=np.argsort(square_dists,axis=1)[:,0:k]     #N2,k
    for i in range(idx.shape[0]):
        final_result[i,:,:]=xyz1[idx[i],:]
    return final_result




def extract_knn_patch(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return k_patches


"""def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    # b = xyz1.get_shape()[0].value
    # n = xyz1.get_shape()[1].value
    # c = xyz1.get_shape()[2].value
    # m = xyz2.get_shape()[1].value
    # xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    # xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    xyz1 = tf.expand_dims(xyz1, axis=1)
    xyz2 = tf.expand_dims(xyz2, axis=2)
    dist = tf.reduce_sum((xyz1 - xyz2) ** 2, -1)

    # outi, out = select_top_k(k, dist)
    # idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    # val = tf.slice(out, [0,0,0], [-1,-1,k])

    val, idx = tf.nn.top_k(-dist, k=k)  # ONLY SUPPORT CPU
    return val, idx"""


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
