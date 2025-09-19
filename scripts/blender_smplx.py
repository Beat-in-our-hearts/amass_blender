import bpy
import numpy as np
from mathutils import Vector, Matrix, Euler
import os
import math

# get context
scene = bpy.context.scene

armature_name_list = ['SMPLX-neutral.halfsquat', 'SMPLX-neutral.legswing', 'SMPLX-neutral.armswing']
armature_name = armature_name_list[2]
armature_obj = bpy.data.objects[armature_name]

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
# define 3dof face
face_names = ['jaw', 'left_eye_smplhf', 'right_eye_smplhf']
bone_names = body_names + hand_names + face_names

# define frame
start_frame = {
    'SMPLX-neutral.halfsquat': 4,
    'SMPLX-neutral.legswing': 4, 
    'SMPLX-neutral.armswing': 4,
}

end_frame = {
    'SMPLX-neutral.halfsquat': 2935,
    'SMPLX-neutral.legswing': 1400, 
    'SMPLX-neutral.armswing': 570,
}

num_frames = end_frame[armature_name] - start_frame[armature_name] + 1

root_orient = np.zeros((num_frames, 3), dtype=np.float32) # 根骨骼旋转
trans = np.zeros((num_frames, 3), dtype=np.float32) # 全局平移
poses = np.zeros((num_frames, 162), dtype=np.float32) # 姿态 (63 + 90 + 9)
pose_body = np.zeros((num_frames, 63), dtype=np.float32) # 身体姿态 (21 joints * 3)
pose_hand = np.zeros((num_frames, 90), dtype=np.float32) # 手部姿态 (30 joints * 3)
pose_face = np.zeros((num_frames, 9), dtype=np.float32) # 面部姿态 (3 joints * 3)

for frame_id in range(start_frame[armature_name], end_frame[armature_name]+1):
    if frame_id % 100 == 0:
        print(f"Processing frame {frame_id}/{num_frames}")
        
    scene.frame_set(frame_id)
    bpy.context.view_layer.update()
    
    # root_orient
    pelvis_bone = armature_obj.pose.bones['pelvis']
    pelvis_global_matrix = armature_obj.matrix_world @ pelvis_bone.matrix
    rotation_matrix = pelvis_global_matrix.to_3x3().normalized()
    rotation_quaternion = rotation_matrix.to_quaternion()
    axis, angle = rotation_quaternion.to_axis_angle()
    root_orient[frame_id - start_frame[armature_name]] = axis * angle 
    
    # trans
    trans[frame_id - start_frame[armature_name]] = pelvis_global_matrix.to_translation()
    
    # poses
    for bone_name in bone_names[1:]:  # skip pelvis
        pose_bone = armature_obj.pose.bones[bone_name]
        if pose_bone.parent:
            local_matrix = pose_bone.parent.matrix.inverted() @ pose_bone.matrix
        else:
            local_matrix = pose_bone.matrix
        local_rotation_matrix = local_matrix.to_3x3().normalized()
        local_rotation_quaternion = local_rotation_matrix.to_quaternion()
        axis, angle = local_rotation_quaternion.to_axis_angle()
        axis_angle = axis * angle
        joint_index = bone_names.index(bone_name) - 1
        poses[frame_id - start_frame[armature_name], joint_index*3:joint_index*3+3] = [axis_angle.x, axis_angle.y, axis_angle.z]

# split poses into body, hand, face
pose_body = poses[:, :63]
pose_hand = poses[:, 63:153]
pose_face = poses[:, 153:]

print(f"{root_orient.shape=}")
print(f"{trans.shape=}")
print(f"{poses.shape=}")
print(f"{pose_body.shape=}")
print(f"{pose_hand.shape=}")
print(f"{pose_face.shape=}")

betas = np.zeros((16,), dtype=np.float32)
smplx_data = {
    'gender': 'neutral',  
    'surface_model_type': 'smplx',
    'mocap_frame_rate': 60,
    'betas': betas,
    'root_orient': root_orient,  
    'trans': trans,
    'poses': poses,
    'pose_body': pose_body, 
    'pose_hand': pose_hand,
    'pose_jaw': pose_face,
    'pose_eye': pose_face,
}

# save to npz
home_dir = os.path.expanduser("~")
documents_dir = os.path.join(home_dir, "Documents")
save_dir = os.path.join(documents_dir, "SMPLX_motion_data")
os.makedirs(save_dir, exist_ok=True)
armature_name = armature_name.replace('.', '-')
output_path = os.path.join(save_dir, f"{armature_name}.npz")
np.savez(output_path, **smplx_data)