import os
import argparse


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio') 
    parser.add_argument('--off_path', type=str, default='/home/siyu_ren/pugeo_pytorch_data/test_mesh/', help='mesh file path') 

    arg = parser.parse_args()

    xyz_file_path=os.path.join(os.path.dirname(__file__),'..','PUGEOx%d/'%arg.up_ratio)
    off_file_path=arg.off_path


    os.system('cd %s'%xyz_file_path)

    off_list=os.listdir(off_file_path)

    for shape in off_list:
        os.system('%s %s %s'%(os.path.join(os.path.dirname(__file__),'p2f/evaluation'),os.path.join(off_file_path,shape),os.path.join(xyz_file_path,shape[:-4]+'.xyz')))


