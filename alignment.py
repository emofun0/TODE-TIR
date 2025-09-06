import os
import glob
import json
import numpy as np
import sys
from PIL import Image
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def project_tir_to_rgb(seq_folder: str, frame_idx: str, bit_depth: str) -> np.ndarray:
    extr = json.load(open(os.path.join(seq_folder, 'extrinsic.json')))
    T_EE2cam_R = np.array(extr['T_EE2cam_R']).reshape(4, 4)
    T_EE2cam_T = np.array(extr['T_EE2cam_T']).reshape(4, 4)
    T_T2R = np.linalg.inv(T_EE2cam_R) @ T_EE2cam_T

    cam_T_info = json.load(open(os.path.join(seq_folder, 'cam_T', 'camera_info.json')))
    K_T = np.array(cam_T_info['intrinsic']).reshape(3, 3)
    cam_R_info = json.load(open(os.path.join(seq_folder, 'cam_R', 'camera_info.json')))
    K_R = np.array(cam_R_info['intrinsic']).reshape(3, 3)

    tir_path = os.path.join(seq_folder, 'cam_T', bit_depth, f'{frame_idx}.png')
    tir_img = np.array(Image.open(tir_path))
    depth_path = os.path.join(seq_folder, 'cam_T', 'depth', 'rendered', f'{frame_idx}.png')
    depth_T = np.array(Image.open(depth_path)).astype(np.float32) / 1000.0

    hT, wT = tir_img.shape[:2]
    uT, vT = np.meshgrid(np.arange(wT), np.arange(hT))
    uv1 = np.stack([uT, vT, np.ones_like(uT)], axis=0).reshape(3, -1)
    depths = depth_T.flatten()
    valid = depths > 0
    uv1v = uv1[:, valid]
    dv = depths[valid]
    pts_T = np.linalg.inv(K_T) @ (uv1v * dv)
    pts1 = np.vstack([pts_T, np.ones((1, pts_T.shape[1]))])

    pts_R = T_T2R @ pts1
    uvz = K_R @ pts_R[:3]
    uv = (uvz[:2] / uvz[2]).T
    tir_vals = tir_img.flatten()[valid]

    rgb_path = os.path.join(seq_folder, 'cam_R', 'rgb', f'{frame_idx}.png')
    hR, wR = np.array(Image.open(rgb_path)).shape[:2]
    grid_x, grid_y = np.mgrid[0:wR, 0:hR]
    aligned = griddata(uv, tir_vals, (grid_x, grid_y), method='nearest', fill_value=0).T
    return aligned

def process_single_frame(frame, seq_folder, out_folder):
    tir8  = project_tir_to_rgb(seq_folder, frame, '8bit')
    tir14 = project_tir_to_rgb(seq_folder, frame, '14bit')
    rgb   = np.array(Image.open(os.path.join(seq_folder, 'cam_R', 'rgb', f'{frame}.png')))
    depth = (np.array(Image.open(
                os.path.join(seq_folder, 'cam_R', 'depth', 'fixed', f'{frame}.png')
             )).astype(np.float32) / 1000.0)
    depth_gt = (np.array(Image.open(
        os.path.join(seq_folder, 'cam_R', 'depth', 'rendered', f'{frame}.png')
    )).astype(np.float32) / 1000.0)

    # 创建输出子文件夹：processed_seq_H_01/000001/
    frame_out_dir = os.path.join(out_folder, frame)
    os.makedirs(frame_out_dir, exist_ok=True)

    # 保存 .npz 文件：processed_seq_H_01/000001/000001.npz
    npz_path = os.path.join(frame_out_dir, f'{frame}.npz')
    np.savez_compressed(npz_path,
                        tir8=tir8, tir14=tir14,
                        rgb=rgb, depth=depth, depth_gt=depth_gt)

    # 拷贝 mask 到：processed_seq_H_01/000001/mask/
    mask_input_dir = os.path.join(seq_folder, 'cam_R', 'mask')
    mask_output_dir = os.path.join(frame_out_dir, 'mask')
    os.makedirs(mask_output_dir, exist_ok=True)

    mask_pattern = os.path.join(mask_input_dir, f'{frame}_*.png')
    mask_paths = sorted(glob.glob(mask_pattern))
    for path in mask_paths:
        fname = os.path.basename(path)
        Image.open(path).save(os.path.join(mask_output_dir, fname))

def create_processed_dataset(seq_folder: str, out_folder: str, mode: str = "single"):
    os.makedirs(out_folder, exist_ok=True)
    pattern = os.path.join(seq_folder, 'cam_T', '8bit', '*.png')
    frames = sorted(os.path.basename(p).split('.')[0] for p in glob.glob(pattern))

    if mode == "multi":
        num_workers = max(1, cpu_count() - 1)
        print(f"Using {num_workers} parallel workers...")
        with Pool(num_workers) as pool:
            list(tqdm(pool.imap_unordered(partial(process_single_frame,
                                                  seq_folder=seq_folder,
                                                  out_folder=out_folder),
                                          frames),
                      total=len(frames), desc="Processing frames"))
    else:
        for frame in tqdm(frames, desc="Processing frames"):
            process_single_frame(frame, seq_folder, out_folder)

def visualize_processed_frame(out_folder: str, frame_idx: str):
    frame_dir = os.path.join(out_folder, frame_idx)
    data = np.load(os.path.join(frame_dir, f'{frame_idx}.npz'), allow_pickle=True)

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0,0].imshow(data['rgb']);        axs[0,0].set_title('RGB');        axs[0,0].axis('off')
    axs[0,1].imshow(data['depth'], cmap='viridis'); axs[0,1].set_title('Depth');     axs[0,1].axis('off')
    axs[0,2].imshow(data['depth_gt'], cmap='viridis'); axs[0,2].set_title('Depth GT'); axs[0,2].axis('off')
    axs[1,0].imshow(data['tir8'], cmap='hot');     axs[1,0].set_title('TIR 8-bit');  axs[1,0].axis('off')
    axs[1,1].imshow(data['tir14'], cmap='hot');    axs[1,1].set_title('TIR 14-bit'); axs[1,1].axis('off')

    # 读取并可视化第一个 mask（如果有）
    mask_dir = os.path.join(frame_dir, 'mask')
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
    if mask_files:
        mask_img = np.array(Image.open(mask_files[0]))
        axs[1,2].imshow(mask_img, cmap='gray')
        axs[1,2].set_title('First Mask')
    else:
        axs[1,2].axis('off')
        axs[1,2].set_title('No Mask Found')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['single', 'multi']:
        print("Usage: python alignment.py [single|multi]")
        sys.exit(1)

    mode = sys.argv[1]

    base_input_root = '/home/data/qiuguanhe/TRansPose/sequences'
    base_output_root = '/home/data/qiuguanhe/TRansPose/sequences'

    for i in range(1, 10):  # from 1 to 9
        seq_name = f'seq_H_0{i}'
        seq_folder = os.path.join(base_input_root, seq_name)
        out_folder = os.path.join(base_output_root, f'processed_{seq_name}')

        print(f'\n▶ Processing {seq_name} in {mode} mode')
        create_processed_dataset(seq_folder, out_folder, mode)

        # 可视化第一帧（如果存在）
        # first_frame = '000001'
        # if os.path.exists(os.path.join(out_folder, first_frame, f'{first_frame}.npz')):
        #     print(f'  ▶ Visualizing first frame of {seq_name}')
        #     visualize_processed_frame(out_folder, first_frame)

