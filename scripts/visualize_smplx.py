from ipywidgets import interact_manual
from ipywidgets import IntSlider

import torch
import numpy as np

from human_body_prior.tools.omni_tools import copy2cpu as c2c
import os
from tqdm import trange
import imageio

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image

from human_body_prior.body_model.body_model import BodyModel

body_type = 'smplx'  # 'smplx' or 'smplh' or 'smpl'

body_model_dict = { 
    "smpl": {"neutral": "SMPL_NEUTRAL.npz", "male": "SMPL_MALE.npz","female": "SMPL_FEMALE.npz"},
    "smplh": {"neutral": "SMPLH_NEUTRAL.npz", "male": "SMPLH_MALE.npz", "female": "SMPLH_FEMALE.npz"},
    "smplx": {"neutral": "SMPLX_NEUTRAL.npz", "male": "SMPLX_MALE.npz", "female": "SMPLX_FEMALE.npz"}
}

pose_list = {
    "smpl": ['pose_body', 'betas'],
    "smplh": ['pose_body', 'betas', 'pose_hand'],
    "smplx": ['pose_body', 'betas', 'pose_hand', 'pose_jaw', 'pose_eye'],
}

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blender_smplx_data_dir = './output/smplx_data/' # the path to blender exported smplx data
amass_npz_fname_list = [os.path.join(blender_smplx_data_dir, f) for f in os.listdir(blender_smplx_data_dir) if f.endswith('_smplx.npz')]
print(f"Available files: {amass_npz_fname_list}")


imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    
for amass_npz_fname in amass_npz_fname_list:
    print(f"Processing file: {amass_npz_fname}")
    
    motion_name = os.path.basename(amass_npz_fname).replace('.npz','')
    bdata = np.load(amass_npz_fname)

    num_betas = 16 # number of body parameters
    num_dmpls = 8 # number of DMPL parameters

    subject_gender = bdata['gender'].item()
    print('Data keys available:%s'%list(bdata.keys()))
    print('The subject of the mocap sequence is  {}.'.format(subject_gender))

    time_length = len(bdata['trans'])
    body_parms = {
        'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
        'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
        'pose_hand': torch.Tensor(bdata['poses'][:, 66:156]).to(comp_device), # controls the finger articulation
        'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
        'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
        'pose_jaw': torch.Tensor(bdata['poses'][:, 156:159]).to(comp_device), # controls the jaw
        'pose_eye': torch.Tensor(bdata['poses'][:, 159:165]).to(comp_device), # controls the eye movement
    }
    print('Body parameter vector shapes: \n{}'.format(' \n'.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
    print('time_length = {}'.format(time_length))

    ## SMPL-X
    bm_smpl_fname = f"./body_models/{body_type}/{body_model_dict[body_type][subject_gender]}"
    bm = BodyModel(bm_fname=bm_smpl_fname, num_betas=num_betas).to(comp_device)

    faces = c2c(bm.f)
    num_verts = bm.init_v_template.shape[1]

    start_frame_idx = 2
    end_frame_idx = time_length-1

    image_list = []
    print({k:v.shape for k,v in body_parms.items() if k in pose_list[body_type]})
    body = bm(**{k:v.to(comp_device) for k,v in body_parms.items() if k in pose_list[body_type]})

    for frame_idx in trange(start_frame_idx, end_frame_idx):
        body_mesh_wofingers = trimesh.Trimesh(vertices=c2c(body.v[frame_idx]), faces=faces, vertex_colors=np.tile(colors['grey'], (num_verts, 1)))
        mv.set_static_meshes([body_mesh_wofingers])
        body_image_wofingers = mv.render(render_wireframe=False)
        image_list.append(body_image_wofingers)
    
    video_path = f'./output/videos/{motion_name}.mp4'
    imageio.mimwrite(video_path, image_list, fps=30)
    print('Video saved to {}'.format(video_path))