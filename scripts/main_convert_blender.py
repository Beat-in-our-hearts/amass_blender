import numpy as np 
import os
from scipy.spatial.transform import Rotation
import json

def load_data(file_dir):
    """加载所有NPZ文件数据"""
    file_list = os.listdir(file_dir)
    name = [f.replace(".npz", "") for f in file_list if f.endswith('.npz')]
    data = {n: np.load(os.path.join(file_dir, f"{n}.npz")) for n in name}
    return data

def rot_matrix_to_axis_angle(rot_mat):
    """
    将3x3旋转矩阵转换为轴角表示
    参数:
        rot_mat: (3, 3) 或 (..., 3, 3) 旋转矩阵
    返回:
        axis_angle: (3,) 或 (..., 3) 轴角表示
    """
    if rot_mat.ndim == 2:
        # 单个矩阵
        rotation = Rotation.from_matrix(rot_mat)
        axis_angle = rotation.as_rotvec()
        return axis_angle
    else:
        # 批量处理
        original_shape = rot_mat.shape[:-2]
        rot_mat_flat = rot_mat.reshape(-1, 3, 3)
        axis_angle_flat = np.array([Rotation.from_matrix(m).as_rotvec() for m in rot_mat_flat])
        return axis_angle_flat.reshape(*original_shape, 3)

def convert_to_smplx_format(data_dict):
    """
    将Blender导出的数据转换为SMPL-X格式
    参数:
        data_dict: 包含全局平移和局部旋转矩阵的字典
    返回:
        smplx_data_dict: 转换为SMPL-X格式的数据字典
    """
    smplx_data_dict = {}
    
    for name, data in data_dict.items():
        global_trans = data['global_translations']  # (N, 3)
        local_rot_mats = data['local_rotations']    # (N, J, 3, 3)
        
        num_frames, num_joints = local_rot_mats.shape[0], local_rot_mats.shape[1]
        
        # 将旋转矩阵转换为轴角格式
        axis_angle_poses = np.zeros((num_frames, num_joints, 3))
        for j in range(num_joints):
            axis_angle_poses[:, j] = rot_matrix_to_axis_angle(local_rot_mats[:, j])
        
        # 重塑为SMPL-X格式: (N, J*3)
        poses = axis_angle_poses.reshape(num_frames, -1)
        
        # 创建SMPL-X格式的数据
        smplx_data = {
            'gender': 'neutral',  # 默认性别
            'surface_model_type': 'smplx',
            'root_orient': poses[:, :3].astype(np.float32),  # 根骨骼旋转
            'pose_body': poses[:, 3:66].astype(np.float32),  # 身体姿态 (21 joints * 3)
            'trans': global_trans.astype(np.float32),
            'betas': np.zeros((10,), dtype=np.float32),  # 默认形状参数
            'poses': poses.astype(np.float32),
            'pose_hand': poses[:, 66:156].astype(np.float32),  # 手部姿态 (30 joints * 3)
            'pose_jaw': poses[:, 156:159].astype(np.float32),   # 下颌姿态 (1 joint * 3)
            'pose_eye': poses[:, 159:165].astype(np.float32),   # 眼睛姿态 (2 joints * 3)
        }
        
        smplx_data_dict[name] = smplx_data
    
    return smplx_data_dict

def save_smplx_data(smplx_data_dict, output_dir):
    """保存SMPL-X格式的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in smplx_data_dict.items():
        output_path = os.path.join(output_dir, f"{name}_smplx.npz")
        np.savez(output_path, **data)
        print(f"Saved SMPL-X format data: {output_path}")

def analyze_motion_data(smplx_data_dict):
    """分析运动数据并生成统计信息"""
    analysis_results = {}
    
    for name, data in smplx_data_dict.items():
        trans = data['trans']
        poses = data['poses']
        
        # 计算统计信息
        stats = {
            'num_frames': len(trans),
            'translation_range': {
                'x': (trans[:, 0].min(), trans[:, 0].max()),
                'y': (trans[:, 1].min(), trans[:, 1].max()),
                'z': (trans[:, 2].min(), trans[:, 2].max())
            },
            'translation_std': {
                'x': trans[:, 0].std(),
                'y': trans[:, 1].std(),
                'z': trans[:, 2].std()
            },
            'pose_variance': poses.std(axis=0).mean(),
            'motion_duration': len(trans) / 30.0  # 假设30fps
        }
        
        analysis_results[name] = stats
    
    return analysis_results

def visualize_smplx_data(smplx_data_dict, output_dir):
    """生成数据可视化（可选，需要matplotlib）"""
    try:
        import matplotlib.pyplot as plt
        
        for name, data in smplx_data_dict.items():
            trans = data['trans']
            
            # 创建轨迹图
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # 2D轨迹图 (X-Z平面)
            ax[0].plot(trans[:, 0], trans[:, 2], 'b-', alpha=0.7)
            ax[0].scatter(trans[0, 0], trans[0, 2], c='green', s=100, label='Start')
            ax[0].scatter(trans[-1, 0], trans[-1, 2], c='red', s=100, label='End')
            ax[0].set_xlabel('X')
            ax[0].set_ylabel('Z')
            ax[0].set_title(f'{name} - Trajectory (X-Z plane)')
            ax[0].legend()
            ax[0].grid(True)
            ax[0].axis('equal')
            
            # 3D位置随时间变化
            time = np.arange(len(trans)) / 30.0  # 假设30fps
            ax[1].plot(time, trans[:, 0], 'r-', label='X')
            ax[1].plot(time, trans[:, 1], 'g-', label='Y')
            ax[1].plot(time, trans[:, 2], 'b-', label='Z')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Position')
            ax[1].set_title(f'{name} - Position over Time')
            ax[1].legend()
            ax[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_visualization.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Generated visualization for {name}")
            
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")

def main():
    # 配置路径
    input_dir = "./data/"  # Blender导出的数据目录
    output_dir = "./output/smplx_data/"  # 处理后的输出目录
    analysis_dir = "./output/analysis/"  # 分析结果目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print("Loading data from Blender export...")
    raw_data = load_data(input_dir)
    
    print("Converting to SMPL-X format...")
    smplx_data = convert_to_smplx_format(raw_data)
    
    print("Saving SMPL-X format data...")
    save_smplx_data(smplx_data, output_dir)
    
    print("Analyzing motion data...")
    analysis_results = analyze_motion_data(smplx_data)
    
    # 保存分析结果
    with open(os.path.join(analysis_dir, "motion_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    print("Generating visualizations...")
    visualize_smplx_data(smplx_data, analysis_dir)
    
    print("Processing completed!")
    
    # 打印简要统计信息
    print("\n=== Motion Statistics ===")
    for name, stats in analysis_results.items():
        print(f"\n{name}:")
        print(f"  Frames: {stats['num_frames']}")
        print(f"  Duration: {stats['motion_duration']:.2f}s")
        print(f"  Translation range X: [{stats['translation_range']['x'][0]:.3f}, {stats['translation_range']['x'][1]:.3f}]")
        print(f"  Translation range Y: [{stats['translation_range']['y'][0]:.3f}, {stats['translation_range']['y'][1]:.3f}]")
        print(f"  Translation range Z: [{stats['translation_range']['z'][0]:.3f}, {stats['translation_range']['z'][1]:.3f}]")

if __name__ == "__main__":
    main()