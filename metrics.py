import os
from tqdm import tqdm
import point_cloud_utils as pcu
from utils import sample_pc
import argparse

# prepare augments
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)  # directory of dense meshes
parser.add_argument('--output_dir', type=str) # directory of generated meshes
args = parser.parse_args()


def main(sample_dir, ref_dir, pc_num=1024):
    print(sample_dir, ref_dir)
    mesh_list = [name for name in os.listdir(ref_dir) if name.endswith('.obj')]
    
    hausdorff_dists, chamfer_dists = [], []
    for mesh_name in tqdm(mesh_list):
        try:
            # sample point cloud from input
            uid = os.path.splitext(mesh_name)[0]
            ref_path = os.path.join(ref_dir, uid + '.obj')
            sample_path = os.path.join(sample_dir, uid + '.obj')
            sample, ref = sample_pc(sample_path, pc_num), sample_pc(ref_path, pc_num)
            
            # compute hausdorff and chamfer distance
            hausdorff_dist = pcu.hausdorff_distance(sample, ref)
            chamfer_dist = pcu.chamfer_distance(sample, ref)
            hausdorff_dists.append(hausdorff_dist)
            chamfer_dists.append(chamfer_dist)
        except Exception as e:
            print(e)
    
    print('hausdorff distance:', sum(hausdorff_dists) / len(hausdorff_dists))
    print('chamfer distance:', sum(chamfer_dists) / len(chamfer_dists))


main(args.input_dir, args.output_dir)
