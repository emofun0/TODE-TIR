import os
import glob
import json
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from utils.data_preparation import process_data, process_data_transpose


class Transpose(Dataset):
    """
    Transpose dataset.
    """

    def __init__(self, data_dir, split, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        data_dir: str, required, the data path;

        split: str in ['train', 'test'], optional, default: 'train', the dataset split option.
        """
        super(Transpose, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.root_dir = data_dir
        self.split = split
        self.split_ratio = kwargs.get('split_ratio', 0.6)
        
        # data_dir = os.path.expanduser("~/data/TRansPose")
        path = os.path.join(data_dir, "sequences")
        print("path",path)
        
        all_seq_dirs = sorted(glob.glob(os.path.join(path, "processed_seq_H_*")))

        print(all_seq_dirs)
        all_frame_dirs = []
        for seq_dir in all_seq_dirs:
            frames = sorted(os.listdir(seq_dir))
            for frame in frames:
                full_path = os.path.join(seq_dir, frame)
                if os.path.isdir(full_path):
                    all_frame_dirs.append(full_path)

        split_idx = int(len(all_frame_dirs) * self.split_ratio) + 1
        if split == 'train':
            self.samples = all_frame_dirs[:split_idx]
        else:
            self.samples = all_frame_dirs[split_idx:]
        print("split_idx", split_idx)
        self.cam_intrinsics_path = os.path.join(self.root_dir, 'sequences/seq_H_01/cam_R','camera_info.json')
        with open(self.cam_intrinsics_path, 'r') as f:
            intr = json.load(f)["intrinsic"]

        self.camera_intrinsic = np.array(intr, dtype=np.float32).reshape(3, 3)

        self.use_aug = kwargs.get('use_augmentation', True)
        self.rgb_aug_prob = kwargs.get('rgb_augmentation_probability', 0.8)
        self.image_size = kwargs.get('image_size', (640, 480))
        print(self.image_size)
        self.depth_min = kwargs.get('depth_min', 0.3)
        self.depth_max = kwargs.get('depth_max', 1.5)
        self.depth_norm = kwargs.get('depth_norm', 1.0)
        self.use_depth_aug = kwargs.get('use_depth_augmentation', True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_dir = self.samples[idx]
        frame_id = os.path.basename(frame_dir)

        npz = np.load(os.path.join(frame_dir, f"{frame_id}.npz"))
        rgb = np.array(npz['rgb'], dtype = np.float32)
        tir8 = np.array(npz['tir8'], dtype = np.float32)
        tir14 = np.array(npz['tir14'], dtype = np.float32)
        depth = np.array(npz['depth'], dtype = np.float32)
        depth_gt = np.array(npz['depth_gt'], dtype = np.float32)
        # 将 ndarray 转成 PIL Image 以便后续 transform
        # rgb = Image.fromarray(rgb)
        # tir8 = Image.fromarray((tir8 / np.max(tir8) * 255).astype(np.uint8))
        # tir14 = Image.fromarray((tir14 / np.max(tir14) * 255).astype(np.uint8))
        # depth = Image.fromarray(depth)  # 如果需要也可以做归一化
        # depth_gt = Image.fromarray(depth_gt)


        # 加载 masks
        mask_dir = os.path.join(frame_dir, 'mask')
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        masks = []
        # 初始化合并后的 depth_gt_mask，与 depth_gt 同样大小
        depth_gt_mask = np.zeros(depth_gt.shape, dtype=np.uint8)

        for m in mask_files:
            mask = Image.open(m)
            mask_array = np.array(mask)
            masks.append(torch.from_numpy(mask_array).long())

            # 将当前物体的 mask 合并到 depth_gt_mask 中
            # 假设 mask 中非零值表示物体区域
            object_mask = (mask_array > 0).astype(np.uint8)
            object_mask_gray = np.any(object_mask > 0, axis=2).astype(np.uint8)
            depth_gt_mask = np.logical_or(depth_gt_mask, object_mask_gray).astype(np.uint8)

        # 没提供 transform 时，默认转 Tensor
        # rgb = torch.from_numpy(np.array(rgb).transpose(2, 0, 1)).float() / 255.0  # [C,H,W]
        # tir8 = torch.from_numpy(np.array(tir8)).unsqueeze(0).float() / 255.0  # [1,H,W]
        # tir14 = torch.from_numpy(np.array(tir14)).unsqueeze(0).float() / 255.0
        # depth = torch.from_numpy(np.array(depth)).unsqueeze(0).float() / 255.0
        # depth_gt = torch.from_numpy(np.array(depth_gt)).unsqueeze(0).float() / 255.0
        if len(mask_files) == 0:
            depth_gt_mask = np.zeros(depth_gt.shape, dtype=np.uint8)

        rgb = cv2.resize(rgb, self.image_size, interpolation = cv2.INTER_LINEAR)
        tir8 = cv2.resize(tir8, self.image_size, interpolation = cv2.INTER_LINEAR)
        depth = cv2.resize(depth, self.image_size, interpolation = cv2.INTER_NEAREST)
        depth_gt = cv2.resize(depth_gt, self.image_size, interpolation=cv2.INTER_NEAREST)




        # tir8 = torch.from_numpy(tir8).unsqueeze(0).float
        # depth = torch.from_numpy(depth).unsqueeze(0).float
        # depth_gt = torch.from_numpy(depth_gt).unsqueeze(0).float
        depth_gt_mask = cv2.resize(depth_gt_mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        neg_zero_mask = np.where(depth_gt < 0.01, 255, 0).astype(np.uint8)
        neg_zero_mask[neg_zero_mask != 0] = 1
        zero_mask = np.logical_not(neg_zero_mask)

        sample = {
            "frame_id": frame_id,
            "rgb": torch.FloatTensor(rgb),
            "tir8": torch.FloatTensor(tir8).unsqueeze(-1),
            "tir14": torch.FloatTensor(tir14).unsqueeze(-1),
            "depth": torch.FloatTensor(depth).unsqueeze(-1),
            "depth_gt": torch.FloatTensor(depth_gt).unsqueeze(-1),
            'zero_mask': torch.BoolTensor(zero_mask),
            "loss_mask": torch.BoolTensor(zero_mask),
            "fx": torch.tensor(603.7765),
            "fy": torch.tensor(604.6316),
            "cx": torch.tensor(329.2594),
            "cy": torch.tensor(246.4848),
            "masks": masks
        }

        data_dict = process_data_transpose(rgb=rgb, depth=depth, depth_gt=depth_gt, depth_gt_mask=depth_gt_mask,
                                 camera_intrinsics=self.camera_intrinsic, scene_type="cluttered",
                                 camera_type=0, split=self.split,
                            image_size=self.image_size, depth_min=self.depth_min, depth_max=self.depth_max,
                            depth_norm=self.depth_norm, use_aug=self.use_aug, rgb_aug_prob=self.rgb_aug_prob,
                            use_depth_aug=self.use_depth_aug)

        data_dict['tir8'] = torch.FloatTensor(tir8)
        data_dict['tir14'] = torch.FloatTensor(tir14)
        data_dict['depth'] = torch.FloatTensor(depth)
        data_dict['depth_gt'] = torch.FloatTensor(depth_gt)

        data_dict['frame_id'] = torch.FloatTensor([float(frame_id)])
        return data_dict
