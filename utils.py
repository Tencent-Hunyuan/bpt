import trimesh
import numpy as np
from x_transformers.autoregressive_wrapper import top_p, top_k


class Dataset:
    '''
    A toy dataset for inference
    '''
    def __init__(self, input_type, input_list):
        super().__init__()
        self.data = []
        if input_type == 'pc_normal':
            for input_path in input_list:
                # load npy
                cur_data = np.load(input_path)
                # sample 4096
                assert cur_data.shape[0] >= 4096, "input pc_normal should have at least 4096 points"
                idx = np.random.choice(cur_data.shape[0], 4096, replace=False)
                cur_data = cur_data[idx]
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})

        elif input_type == 'mesh':
            mesh_list, pc_list = [], []
            for input_path in input_list:
                # sample point cloud and normal from mesh
                cur_data = trimesh.load(input_path, force='mesh')
                cur_data = apply_normalize(cur_data)
                mesh_list.append(cur_data)
                pc_list.append(sample_pc(cur_data, pc_num=4096, with_normal=True))

            for input_path, cur_data in zip(input_list, pc_list):
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})
                
        print(f"dataset total data samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict['pc_normal'] = self.data[idx]['pc_normal']
        data_dict['uid'] = self.data[idx]['uid']

        return data_dict
    

def joint_filter(logits, k = 50, p=0.95):
    logits = top_k(logits, k = k)
    logits = top_p(logits, thres = p)
    return logits


def apply_normalize(mesh):
    '''
    normalize mesh to [-1, 1]
    '''
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale * 2 * 0.95)

    return mesh



def sample_pc(mesh_path, pc_num, with_normal=False):

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    mesh = apply_normalize(mesh)
    
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points

    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    
    return pc_normal


