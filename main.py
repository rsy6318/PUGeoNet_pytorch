import os
import torch
import dataloader as provider
import tensorflow as tf
import numpy as np
import argparse
import model
import loss
import logging
from glob import glob
from pc_util import extract_knn_patch, normalize_point_cloud, farthest_point_sample, group_points
from datetime import datetime
from tqdm import tqdm, trange
print(torch.cuda.is_available())

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()


def eval_patches(xyz, arg, model):
    centroids = farthest_point_sample(xyz, arg.num_patch)

    patches = group_points(xyz, centroids, arg.num_point)

    normalized_patches, patch_centroid, furthest_distance = normalize_point_cloud(patches)

    dense_patches_list = []
    dense_normal_list = []
    sparse_normal_list = []

    for i in range(normalized_patches.shape[0]):
        dense_patches, dense_normal, sparse_normal = model(
            torch.from_numpy(normalized_patches[i]).unsqueeze(0).transpose(1, 2), 1)
        dense_patches_list.append(dense_patches.transpose(1, 2).cpu().numpy())
        dense_normal_list.append(dense_normal.transpose(1, 2).cpu().numpy())
        sparse_normal_list.append(sparse_normal.transpose(1, 2).cpu().numpy())

    gen_ddense_xyz = np.concatenate(dense_patches_list, axis=0)
    gen_ddense_xyz = (gen_ddense_xyz * furthest_distance) + patch_centroid
    gen_ddense_normal = np.concatenate(dense_normal_list, axis=0)

    return np.reshape(gen_ddense_xyz, (-1, 3)), np.reshape(gen_ddense_normal, (-1, 3))


def evaluate(model, arg):
    model.eval()
    shapes = glob(arg.eval_xyz + '/*.xyz')

    for i, item in enumerate(shapes):
        obj_name = item.split('/')[-1]
        data = np.loadtxt(item)
        input_sparse_xyz = data[:, 0:3]
        input_sparse_normal = data[:, 3:6]
        normalize_sparse_xyz, centroid, furthest_distance = normalize_point_cloud(input_sparse_xyz)
        dense_xyz, dense_normal = eval_patches(normalize_sparse_xyz, arg)
        dense_xyz = dense_xyz * furthest_distance + centroid
        gen_dense = np.concatenate((dense_xyz, dense_normal), axis=-1)
        path = os.path.join(arg.eval_path, obj_name)
        np.savetxt(path, gen_dense)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_record', default='D:\\PUGEO\\tfrecord_x4_normal\\*.tfrecord', help='data path')
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio')  #
    parser.add_argument('--model', default='model_pugeo', help='Model for upsampling')
    parser.add_argument('--num_point', type=int, default=256, help='Point Number')

    # for phase train
    #parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training')
    parser.add_argument('--max_epoch', type=int, default=600, help='Epoch to run')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--reg_normal1', type=float, default=0.1)
    parser.add_argument('--reg_normal2', type=float, default=0.1)
    parser.add_argument('--jitter_sigma', type=float, default=0.01)
    parser.add_argument('--jitter_max', type=float, default=0.03)

    # for phase test
    parser.add_argument('--pretrained', default='', help='Model stored')
    parser.add_argument('--eval_xyz', default='D:\\PUGEO\\test_5000', help='Folder to evaluate')
    parser.add_argument('--num_shape_point', type=int, default=5000, help='Point Number per shape')
    parser.add_argument('--patch_num_ratio', type=int, default=3, help='Number of points covered by patch')
    arg = parser.parse_args()
    arg.num_patch = int(arg.num_shape_point / arg.num_point * arg.patch_num_ratio)
    arg.log_dir='log_x%d'%arg.up_ratio
    try:
        os.mkdir(arg.log_dir)
    except:
        pass
    global LOG_FOUT
    LOG_FOUT = open(os.path.join(arg.log_dir, 'log.txt'), 'w')
    LOG_FOUT.write(str(datetime.now()) + '\n')
    LOG_FOUT.write(os.path.abspath(__file__) + '\n')
    LOG_FOUT.write(str(arg) + '\n')

    dataloader = provider.Fetcher(arg.train_record, batch_size=arg.batch_size, step_ratio=arg.up_ratio,
                                  up_ratio=arg.up_ratio, num_in_point=arg.num_point,
                                  num_shape_point=arg.num_shape_point, jitter=True, drop_out=1.0,
                                  jitter_max=arg.jitter_max, jitter_sigma=arg.jitter_sigma)

    sess=tf.Session()
    dataloader.initialize(sess,arg.up_ratio,False)

    model=model.pugeonet(up_ratio=arg.up_ratio,knn=30)
    model=model.cuda()

    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    current_lr=arg.learning_rate

    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)

    for epoch in range(arg.max_epoch):
        loss_sum_all=[]
        loss_sum_dense_cd = []
        loss_sum_dense_normal = []
        loss_sum_sparse_normal = []
        for batch_idx in tqdm(range(dataloader.num_batches),desc='epoch %d'%epoch):
            input_sparse, gt_dense, input_r, _ = dataloader.fetch(sess)
            input_sparse_xyz = input_sparse[:, :, 0:3]
            input_sparse_normal = input_sparse[:, :, 3:6]
            sparse_l2 = np.linalg.norm(input_sparse_normal, axis=-1, keepdims=True)

            gt_dense_xyz = gt_dense[:, :, 0:3]
            gt_dense_normal = gt_dense[:, :, 3:6]
            dense_l2 = np.linalg.norm(gt_dense_normal, axis=-1, keepdims=True)

            input_sparse_xyz = torch.from_numpy(input_sparse_xyz).cuda().transpose(1,2)
            input_sparse_normal = torch.from_numpy(input_sparse_normal).cuda().transpose(1,2)
            gt_dense_xyz = torch.from_numpy(gt_dense_xyz).cuda().transpose(1,2)
            gt_dense_normal = torch.from_numpy(gt_dense_normal).cuda().transpose(1,2)
            input_r = torch.from_numpy(input_r).cuda()

            optimizer.zero_grad()

            model.train()
            gen_dense_xyz,gen_dense_normal,gen_sparse_normal=model(input_sparse_xyz, bradius=input_r)

            loss_dense_cd,cd_idx1,cd_idx2=loss.cd_loss(gen_dense_xyz,gt_dense_xyz,input_r)
            loss_dense_normal=loss.abs_dense_normal_loss(gen_dense_normal,gt_dense_normal,cd_idx1,cd_idx2,input_r)
            loss_sparse_normal=loss.abs_sparse_normal_loss(gen_sparse_normal,input_sparse_normal,input_r)

            '''print(loss_dense_cd.size())
            print(loss_dense_normal.size())
            print(loss_sparse_normal.size())'''

            loss_all = 100 * loss_dense_cd + arg.reg_normal1 * loss_dense_normal + arg.reg_normal2 * loss_sparse_normal

            loss_all.backward()
            optimizer.step()

            loss_sum_all.append(loss_all.detach().cpu().numpy())
            loss_sum_dense_cd.append(loss_dense_cd.detach().cpu().numpy())
            loss_sum_dense_normal.append(loss_dense_normal.detach().cpu().numpy())
            loss_sum_sparse_normal.append(loss_sparse_normal.detach().cpu().numpy())

            '''log_string('epoch: %d total loss: %f, cd: %f, dense normal: %f, sparse normal: %f\n' % (
                epoch, loss_all.detach().cpu().numpy(), loss_dense_cd.detach().cpu().numpy(),
                loss_dense_normal.detach().cpu().numpy(),
                loss_sparse_normal.detach().cpu().numpy()))'''
            '''print('epoch: %d total loss: %0.7f, cd: %0.7f, dense normal: %0.7f, sparse normal: %0.7f' % (
                epoch, loss_all.detach().cpu().numpy(), loss_dense_cd.detach().cpu().numpy(),
                loss_dense_normal.detach().cpu().numpy(),
                loss_sparse_normal.detach().cpu().numpy()))'''

        loss_sum_all = np.asarray(loss_sum_all)
        loss_sum_dense_cd = np.asarray(loss_sum_dense_cd)
        loss_sum_dense_normal = np.asarray(loss_sum_dense_normal)
        loss_sum_sparse_normal = np.asarray(loss_sum_sparse_normal)
        log_string('epoch: %d total loss: %f, cd: %f, dense normal: %f, sparse normal: %f\n' % (
                    epoch, round(loss_sum_all.mean(), 7), round(loss_sum_dense_cd.mean(), 7), round(loss_sum_dense_normal.mean(), 7),
                    round(loss_sum_sparse_normal.mean(), 7)))
        if epoch%10==0:
            torch.save(model.state_dict(),os.path.join(arg.log_dir,'model_%d.t7'%epoch))

        if epoch%20==0 and epoch>0:
            current_lr = current_lr * 0.8
            if current_lr < arg.min_lr:
                current_lr = arg.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
    torch.save(model.state_dict(), os.path.join(arg.log_dir, 'model.t7'))