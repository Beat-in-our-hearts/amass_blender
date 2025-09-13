# blender script to extract global translations and local rotations from armature animations
import bpy
import numpy as np
from mathutils import Matrix
import os

# get context
scene = bpy.context.scene

# define armature and get objects
armature_name_list = ['SMPLX-neutral.halfsquat', 'SMPLX-neutral.legswing', 'SMPLX-neutral.armswing']
armature_obj_list = {armature_name: bpy.data.objects[armature_name] for armature_name in armature_name_list}

# define 22dof body names
body_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 
              'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 
              'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']

# define 30dof hand names
hand_names = ['left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 
              'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 
              'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 
              'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 
              'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3']

# define face
face_names = ['jaw', 'left_eye_smplhf', 'right_eye_smplhf']

bone_names = body_names + hand_names + face_names

num_joints = len(bone_names)

# define frame
start_frame = {
    'SMPLX-neutral.halfsquat': 0,
    'SMPLX-neutral.legswing': 0, 
    'SMPLX-neutral.armswing': 0,
}

end_frame = {
    'SMPLX-neutral.halfsquat': 2939,
    'SMPLX-neutral.legswing': 1677, 
    'SMPLX-neutral.armswing': 573,
}

num_frames = {
    'SMPLX-neutral.halfsquat': 2939,
    'SMPLX-neutral.legswing': 1677, 
    'SMPLX-neutral.armswing': 573,
}

# init buffers
global_translations = {
    armature_name: np.zeros((num_frames[armature_name], 3), dtype=np.float32) for armature_name in armature_name_list
}
local_rotations = {
    armature_name: np.zeros((num_frames[armature_name], num_joints, 3, 3), dtype=np.float32) for armature_name in armature_name_list
}

max_frames = max(num_frames.values())

for frame_id in range(max_frames):
    if frame_id % 100 == 0:
        print(f"Processing frame {frame_id}/{max_frames}")
    scene.frame_set(frame_id)
    bpy.context.view_layer.update()
    
    for armature_name, armature_obj in armature_obj_list.items():
        
        if frame_id >= num_frames[armature_name]:
            continue
        
        root_bone = armature_obj.pose.bones['pelvis']
        root_matrix = armature_obj.matrix_world @ root_bone.matrix
        global_translations[armature_name][frame_id] = root_matrix.to_translation()
        
        for joint_id, bone_name in enumerate(bone_names):
            pose_bone = armature_obj.pose.bones[bone_name]
            
            if pose_bone.parent:
                local_matrix = pose_bone.parent.matrix.inverted() @ pose_bone.matrix
            else:
                local_matrix = pose_bone.matrix
                
            rot_matrix = local_matrix.to_3x3().normalized()
            local_rotations[armature_name][frame_id, joint_id] = np.array(rot_matrix)
            
print("save all smplx motion data")

# save to npz
home_dir = os.path.expanduser("~")
documents_dir = os.path.join(home_dir, "Documents")
save_dir = os.path.join(documents_dir, "SMPLX_motion_data")
os.makedirs(save_dir, exist_ok=True)

for armature_name in armature_name_list:
    np.savez(os.path.join(save_dir, f"{armature_name}.npy"),
             global_translations=global_translations[armature_name], 
             local_rotations=local_rotations[armature_name])
    print(f"Saved {armature_name} motion data to {save_dir}")