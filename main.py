import yaml
import torch
import os
import argparse
import trimesh
import numpy as np
from model.serializaiton import BPT_deserialize
from model.model import MeshTransformer
from utils import joint_filter, Dataset
from model.data_utils import to_mesh

# prepare arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/BPT-pc-open-8k-8-16.yaml')
parser.add_argument('--model_path', type=str)
parser.add_argument('--input_dir', default=None, type=str)
parser.add_argument('--input_path', default=None, type=str)
parser.add_argument('--out_dir', default="output", type=str)
parser.add_argument('--input_type', choices=['mesh','pc_normal'], default='mesh')
parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.5)  # key sampling parameter
parser.add_argument('--condition', type=str, default='pc')
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # prepare model with fp16 precision
    model = MeshTransformer(
        dim = config['dim'],
        attn_depth = config['depth'],
        max_seq_len = config['max_seq_len'],
        dropout = config['dropout'],
        mode = config['mode'],
        num_discrete_coors= 2**int(config['quant_bit']),
        block_size = config['block_size'],
        offset_size = config['offset_size'],
        conditioned_on_pc = config['conditioned_on_pc'],
        use_special_block = config['use_special_block'],
        encoder_name = config['encoder_name'],
        encoder_freeze = config['encoder_freeze'],
    )
    model.load(args.model_path)
    model = model.eval()
    model = model.half()
    model = model.cuda()
    num_params = sum([param.nelement() for param in model.decoder.parameters()])
    print('Number of parameters: %.2f M' % (num_params / 1e6))
    print(f'Block Size: {model.block_size} | Offset Size: {model.offset_size}')

    # prepare data
    if args.input_dir is not None:
        input_list = sorted(os.listdir(args.input_dir))
        if args.input_type == 'pc_normal':
            # npy file with shape (n, 6):
            # point_cloud (n, 3) + normal (n, 3)
            input_list = [os.path.join(args.input_dir, x) for x in input_list if x.endswith('.npy')]
        else:
            # mesh file (e.g., obj, ply, glb)
            input_list = [os.path.join(args.input_dir, x) for x in input_list]
        dataset = Dataset(args.input_type, input_list)

    elif args.input_path is not None:
        dataset = Dataset(args.input_type, [args.input_path])

    else:
        raise ValueError("input_dir or input_path must be provided.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last = False,
        shuffle = False,
    )

    os.makedirs(args.output_path, exist_ok=True)
    with torch.no_grad():
        for it, data in enumerate(dataloader):
            if args.condition == 'pc':
                # generate codes with model
                codes = model.generate(
                    batch_size = args.batch_size,
                    temperature = args.temperature,
                    pc = data['pc_normal'].cuda().half(),
                    filter_logits_fn = joint_filter,
                    filter_kwargs = dict(k=50, p=0.95),
                    return_codes=True,
                )

            coords = []
            try:
                # decoding codes to coordinates
                for i in range(len(codes)):
                    code = codes[i]
                    code = code[code != model.pad_id].cpu().numpy()
                    vertices = BPT_deserialize(
                        code, 
                        block_size = model.block_size, 
                        offset_size = model.offset_size,
                        use_special_block = model.use_special_block,
                    )
                    coords.append(vertices)
            except:
                coords.append(np.zeros(3, 3))

            # convert coordinates to mesh
            for i in range(args.batch_size):
                uid = data['uid'][i]
                vertices = coords[i]
                faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
                mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
                num_faces = len(mesh.faces)
                # set the color for mesh
                face_color = np.array([120, 154, 192, 255], dtype=np.uint8)
                face_colors = np.tile(face_color, (num_faces, 1))
                mesh.visual.face_colors = face_colors
                mesh.export(f'{args.output_path}/{uid}_mesh.obj')

                # save pc
                if args.condition == 'pc':
                    pcd = data['pc_normal'][i].cpu().numpy()
                    point_cloud = trimesh.points.PointCloud(pcd[..., 0:3])
                    point_cloud.export(f'{args.output_path}/{uid}_pc.ply', "ply")
