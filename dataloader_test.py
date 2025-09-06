import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.TRansPose import Transpose

basic_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

data_root = '/home/data/qiuguanhe/TRansPose'
dataset = Transpose(
    data_dir=data_root,
    split='train',
)

# create dataloader
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

for batch in loader:
    print("\n=== Batch Info ===")
    print("Frame IDs:", batch["frame_id"])
    print("RGB:", batch["rgb"].shape)  # [B, 3, H, W]
    print("TIR8:", batch["tir8"].shape)  # [B, 1, H, W]
    print("TIR14", batch["tir14"].shape)
    print("Depth:", batch["depth"].shape)  # [B, 1, H, W]
    print("fx", batch["fx"])
    print("fy", batch["fy"])
    print("cx", batch["cx"])
    print("cy", batch["cy"])

    # multi-modal input tensor
    input_tensor = torch.cat([batch["rgb"], batch["tir8"], batch["depth"]], dim=1)
    print("Multi-modal input shape:", input_tensor.shape)  # [B, 5, H, W]

    # if batch["depth_gt_mask"]:
    #     print("Number of masks in first sample:", len(batch["masks"]))
    #     print("First mask shape:", batch["masks"][0][0].shape)

