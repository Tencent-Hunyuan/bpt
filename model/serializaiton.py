import trimesh
import numpy as np
from .data_utils import discretize, undiscretize


def patchified_mesh(mesh: trimesh.Trimesh, special_token = -2, fix_orient=True):
    sequence = []
    unvisited = np.full(len(mesh.faces), True)
    degrees = mesh.vertex_degree.copy()

    # with fix_orient=True, the normal would be correct.
    # but this may increase the difficulty for learning.
    if fix_orient:
        face_orient = {}
        for ind, face in enumerate(mesh.faces):
            v0, v1, v2 = face[0], face[1], face[2]
            face_orient['{}-{}-{}'.format(v0, v1, v2)] = True
            face_orient['{}-{}-{}'.format(v1, v2, v0)] = True
            face_orient['{}-{}-{}'.format(v2, v0, v1)] = True
            face_orient['{}-{}-{}'.format(v2, v1, v0)] = False
            face_orient['{}-{}-{}'.format(v1, v0, v2)] = False
            face_orient['{}-{}-{}'.format(v0, v2, v1)] = False

    while sum(unvisited):
        unvisited_faces = mesh.faces[unvisited]
        
        # select the patch center
        cur_face = unvisited_faces[0]
        max_deg_vertex_id = np.argmax(degrees[cur_face])
        max_deg_vertex = cur_face[max_deg_vertex_id]
        
        # find all connected faces
        selected_faces = []
        for face_idx in mesh.vertex_faces[max_deg_vertex]:
            if face_idx != -1 and unvisited[face_idx]:
                face = mesh.faces[face_idx]
                u, v = sorted([vertex for vertex in face if vertex != max_deg_vertex])
                selected_faces.append([u, v, face_idx])
                
        face_patch = set()
        selected_faces = sorted(selected_faces)
        
        # select the start vertex, select it if it only appears once (the start or end), 
        # else select the lowest index
        cnt = {}
        for u, v, _ in selected_faces:
            cnt[u] = cnt.get(u, 0) + 1
            cnt[v] = cnt.get(v, 0) + 1
        starts = []
        for vertex, num in cnt.items():
            if num == 1:
                starts.append(vertex)
        start_idx = min(starts) if len(starts) else selected_faces[0][0]
        
        res = [start_idx]
        while len(res) <= len(selected_faces):
            vertex = res[-1]
            for u_i, v_i, face_idx_i in selected_faces:
                if face_idx_i not in face_patch and vertex in (u_i, v_i):
                    u_i, v_i = (u_i, v_i) if vertex == u_i else (v_i, u_i)
                    res.append(v_i)
                    face_patch.add(face_idx_i)
                    break
            
            if res[-1] == vertex:
                break
            
        if fix_orient and len(res) >= 2 and not face_orient['{}-{}-{}'.format(max_deg_vertex, res[0], res[1])]:
            res = res[::-1]
        
        # reduce the degree of related vertices and mark the visited faces
        degrees[max_deg_vertex] = len(selected_faces) - len(res) + 1
        for pos_idx, vertex in enumerate(res):
            if pos_idx in [0, len(res) - 1]:
                degrees[vertex] -= 1
            else:
                degrees[vertex] -= 2
        for face_idx in face_patch:
            unvisited[face_idx] = False 
        sequence.extend(
            [mesh.vertices[max_deg_vertex]] + 
            [mesh.vertices[vertex_idx] for vertex_idx in res] + 
            [[special_token] * 3]
        )
        
    assert sum(degrees) == 0, 'All degrees should be zero'

    return np.array(sequence)



def get_block_representation(
        sequence, 
        block_size=8, 
        offset_size=16, 
        block_compressed=True, 
        special_token=-2, 
        use_special_block=True
    ):
    '''
    convert coordinates from Cartesian system to block indexes.
    '''
    special_block_base = block_size**3 + offset_size**3
    # prepare coordinates
    sp_mask = sequence != special_token
    sp_mask = np.all(sp_mask, axis=1)
    coords = sequence[sp_mask].reshape(-1, 3)
    coords = discretize(coords)

    # convert [x, y, z] to [block_id, offset_id]
    block_id = coords // offset_size
    block_id = block_id[:, 0] * block_size**2 + block_id[:, 1] * block_size + block_id[:, 2]
    offset_id = coords % offset_size
    offset_id = offset_id[:, 0] * offset_size**2 + offset_id[:, 1] * offset_size + offset_id[:, 2]
    offset_id += block_size**3
    block_coords = np.concatenate([block_id[..., None], offset_id[..., None]], axis=-1).astype(np.int64)
    sequence[:, :2][sp_mask] = block_coords
    sequence = sequence[:, :2]
    
    # convert to codes
    codes = []
    cur_block_id = sequence[0, 0]
    codes.append(cur_block_id)
    for i in range(len(sequence)):
        if sequence[i, 0] == special_token:
            if not use_special_block:
                codes.append(special_token)
            cur_block_id = special_token
            
        elif sequence[i, 0] == cur_block_id:
            if block_compressed:
                codes.append(sequence[i, 1])
            else:
                codes.extend([sequence[i, 0], sequence[i, 1]])
                
        else:
            if use_special_block and cur_block_id == special_token:
                block_id = sequence[i, 0] + special_block_base
            else:
                block_id = sequence[i, 0]
            codes.extend([block_id, sequence[i, 1]])
            cur_block_id = block_id

    codes = np.array(codes).astype(np.int64)
    sequence = codes
    
    return sequence.flatten()


def BPT_serialize(mesh: trimesh.Trimesh):
    # serialize mesh with BPT
    
    # 1. patchify faces into patches
    sequence = patchified_mesh(mesh, special_token=-2)
    
    # 2. convert coordinates to block-wise indexes
    codes = get_block_representation(
        sequence, block_size=8, offset_size=16, 
        block_compressed=True, special_token=-2, use_special_block=True
    )
    return codes    


def decode_block(sequence, compressed=True, block_size=8, offset_size=16):
    
    # decode from compressed representation
    if compressed:
        res = []
        res_block = 0
        for token_id in range(len(sequence)):                
            if block_size**3 + offset_size**3 > sequence[token_id] >= block_size**3:
                res.append([res_block, sequence[token_id]])
            elif block_size**3 > sequence[token_id] >= 0:
                res_block = sequence[token_id]
            else:
                print('[Warning] too large offset idx!', token_id, sequence[token_id])
        sequence = np.array(res)
    
    block_id, offset_id = np.array_split(sequence, 2, axis=-1)
    
    # from hash representation to xyz 
    coords = []
    offset_id -= block_size**3
    for i in [2, 1, 0]:
        axis = (block_id // block_size**i) * offset_size + (offset_id // offset_size**i)
        block_id %= block_size**i
        offset_id %= offset_size**i    
        coords.append(axis)
    
    coords = np.concatenate(coords, axis=-1) # (nf 3)
    
    # back to continuous space
    coords = undiscretize(coords)

    return coords


def BPT_deserialize(sequence, block_size=8, offset_size=16, compressed=True, special_token=-2, use_special_block=True):
    # decode codes back to coordinates

    special_block_base = block_size**3 + offset_size**3
    start_idx = 0
    vertices = []
    for i in range(len(sequence)):
        sub_seq = []
        if not use_special_block and (sequence[i] == special_token or i == len(sequence) - 1):
            sub_seq = sequence[start_idx:i]
            sub_seq = decode_block(sub_seq, compressed=compressed, block_size=block_size, offset_size=offset_size)
            start_idx = i + 1

        elif use_special_block and \
            (special_block_base <= sequence[i] < special_block_base + block_size**3 or i == len(sequence)-1):
            if i != 0:
                sub_seq = sequence[start_idx:i] if i != len(sequence) - 1 else sequence[start_idx: i+1]
                if special_block_base <= sub_seq[0] < special_block_base + block_size**3:
                    sub_seq[0] -= special_block_base
                sub_seq = decode_block(sub_seq, compressed=compressed, block_size=block_size, offset_size=offset_size)
                start_idx = i

        if len(sub_seq):
            center, sub_seq = sub_seq[0], sub_seq[1:]
            for j in range(len(sub_seq) - 1):
                vertices.extend([center.reshape(1, 3), sub_seq[j].reshape(1, 3), sub_seq[j+1].reshape(1, 3)])

    # (nf, 3)
    return np.concatenate(vertices, axis=0) 


if __name__ == '__main__':
    # a simple demo for serialize and deserialize mesh with bpt
    from data_utils import load_process_mesh, to_mesh
    import torch
    mesh = load_process_mesh('/path/to/your/mesh', quantization_bits=7)
    mesh['faces'] = np.array(mesh['faces'])
    mesh = to_mesh(mesh['vertices'], mesh['faces'], transpose=True)
    mesh.export('gt.obj')
    codes = BPT_serialize(mesh)
    coordinates = BPT_deserialize(codes)
    faces = torch.arange(1, len(coordinates) + 1).view(-1, 3)
    mesh = to_mesh(coordinates, faces, transpose=False, post_process=False)
    mesh.export('reconstructed.obj')
