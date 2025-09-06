      
### Modified fine-tune script for TIR-aware TODE model ###

import os
import yaml
import torch
import logging
import argparse
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.constants import LOSS_INF
from utils.functions import display_results, to_device
from time import perf_counter

torch.multiprocessing.set_sharing_strategy('file_system')

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
# Append terminal output to stats/finetune.txt without overwriting existing content
file_handler = logging.FileHandler(os.path.join('stats', 'finetune.txt'), mode='a')  # Append mode
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument('--cfg', '-c', default=os.path.join('configs', 'finetune.yaml'),
                    help='path to the configuration file', type=str)
parser.add_argument('--checkpoint', type=str, default=os.path.join('checkpoints', 'transcg_checkpoint.tar'),
                    help='Path to the RGBD checkpoint for finetuning')
args = parser.parse_args()

cuda_id = "cuda:" + str(args.cuda)

with open(args.cfg, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)
tensorboard_log = builder.get_tensorboard_log()

logger.info('Building TIR fine-tuning model ...')
model = builder.get_model()  # should return TodeTIR with in_chans=5

if builder.multigpu():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

# ----------- load checkpoint from 4-channel model -------------
def load_rgbd_to_tir_finetune(model, checkpoint_path):
    """
    Load pretrained RGBD model weights into TIR model
    The key difference: expand the patch embedding layer to accept 5 channels instead of 4
    """
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    
    

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    new_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                # Shapes match, direct copy
                new_dict[k] = v
            elif 'encoder.patch_embed.proj.weight' in k and model_dict[k].shape[1] == 5:
                # This is the patch embedding layer that needs channel expansion
                # Original: [embed_dim, 4, patch_size, patch_size]
                # Target: [embed_dim, 5, patch_size, patch_size]
                logger.info(f'Expanding {k} from {v.shape} to {model_dict[k].shape}')
                new_weight = model_dict[k].clone()
                new_weight[:, :4, :, :] = v  # Copy RGBD weights
                # The 5th channel (TIR) is initialized with the existing initialization
                new_dict[k] = new_weight
            else:
                logger.warning(f'Shape mismatch for {k}: checkpoint {v.shape} vs model {model_dict[k].shape}')

    # Load the weights
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    logger.info("✅ Loaded RGBD pretrained weights into 5-channel TIR model.")
    logger.info(f"Loaded {len(new_dict)} layers from checkpoint.")


# ----------- freeze encoder + decoder strategically ----------
def freeze_rgbd_encoder_decoder(model):
    """
    Based on TODE architecture, freeze strategically:

    TRAINABLE:
    1. Patch Embedding (Conv Projection) - to learn TIR channel features
    2. Layer Norm layers - adaptive normalization for new modality
    3. Final prediction head - task-specific adaptation
    4. SE (Squeeze-Excitation) layers in decoder - attention mechanism adaptation

    FROZEN:
    1. Transformer layers (W-MSA, SW-MSA, MLP) - spatial reasoning preserved
    2. Most upsampling convolutions - structural feature reconstruction
    """
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        should_train = False

        # 1. Patch Embedding - CRITICAL for new TIR modality
        if 'patch_embed' in name:
            should_train = True
            logger.info(f'Trainable (Patch Embed): {name} - {param.shape}')

        # 2. Layer Normalization - adaptive normalization
        elif 'norm' in name.lower() or 'layer_norm' in name:
            should_train = True
            logger.info(f'Trainable (LayerNorm): {name} - {param.shape}')

        # 3. SE (Squeeze-Excitation) layers - attention adaptation
        elif 'se' in name.lower() and ('down' in name or 'layer' in name):
            should_train = True
            logger.info(f'Trainable (SE Layer): {name} - {param.shape}')

        # 4. Final prediction head - task adaptation
        elif 'final' in name:
            should_train = True
            logger.info(f'Trainable (Final Head): {name} - {param.shape}')

        # 5. Decoder upsampling - partial training for feature fusion
        elif 'decoder' in name and ('up' in name and 'net' in name):
            should_train = True
            logger.info(f'Trainable (Decoder Up): {name} - {param.shape}')

        # FROZEN: Transformer blocks (spatial reasoning)
        elif any(x in name for x in ['attn', 'mlp', 'qkv', 'proj']):
            should_train = False

        # FROZEN: Core encoder transformations
        elif 'encoder' in name and any(x in name for x in ['layers', 'blocks']):
            should_train = False

        # Default: train other parameters (bias terms, etc.)
        else:
            should_train = True
            if param.numel() > 100:  # Only log significant parameters
                logger.info(f'Trainable (Other): {name} - {param.shape}')

        param.requires_grad = should_train
        if should_train:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    logger.info(f'=' * 50)
    logger.info(f'FREEZING STRATEGY SUMMARY:')
    logger.info(
        f'Frozen parameters: {frozen_params:,} ({frozen_params / (frozen_params + trainable_params) * 100:.1f}%)')
    logger.info(
        f'Trainable parameters: {trainable_params:,} ({trainable_params / (frozen_params + trainable_params) * 100:.1f}%)')
    logger.info(f'Total parameters: {frozen_params + trainable_params:,}')
    logger.info(f'=' * 50)


# Load pretrained weights and freeze layers
load_rgbd_to_tir_finetune(model, args.checkpoint)
freeze_rgbd_encoder_decoder(model)

# -------- GPU & multi-gpu ----------
if builder.multigpu():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = nn.DataParallel(model)
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Always try cuda:0 first
    model.to(device)

logger.info(f'Using device: {device}')

# ----------- dataloaders (must include 'tir8') ------------
logger.info('Building TIR dataloaders ...')
train_dataloader = builder.get_dataloader(split='train')
test_dataloader = builder.get_dataloader(split='test')

# Optional: real test data (commented out same as train_transpose.py)
# test_real_dataloader = builder.get_dataloader(dataset_params={"test": {"type": "cleargrasp-syn", "data_dir": "cleargrasp", "image_size": (320, 240),
#                                                                  "use_augmentation": False, "depth_min": 0.0, "depth_max": 10.0, "depth_norm": 1.0}}, split='test')

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = builder.get_max_epoch()
stats_dir = builder.get_stats_dir()

# Check if there's an existing fine-tuning checkpoint
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    
    

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    checkpoint_metrics = checkpoint['metrics']
    checkpoint_loss = checkpoint['loss']
    logger.info("Fine-tuning checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))

logger.info('Building optimizer and learning rate schedulers ...')
resume = (start_epoch > 0)
# Use the learning rate from config (already optimized for fine-tuning in finetune.yaml)
optimizer = builder.get_optimizer(model, resume=resume, resume_lr = 5e-5)
lr_scheduler = builder.get_lr_scheduler(optimizer, resume=resume, resume_epoch=(start_epoch - 1 if resume else None))

criterion = builder.get_criterion()
metrics = builder.get_metrics()


# ----------- train/test loop with TIR ----------------
def train_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    losses = []
    with tqdm(train_dataloader) as pbar:
        for data_dict in pbar:
            optimizer.zero_grad()
            data_dict = to_device(data_dict, device)

            # Use 'tir8' key consistent with train_transpose.py
            res = model(data_dict['rgb'], data_dict['depth'], data_dict['tir8'])
            n, h, w = data_dict['depth'].shape
            data_dict['pred'] = res.view(n, h, w)

            loss_dict = criterion(data_dict)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()

            if 'smooth' in loss_dict.keys():
                pbar.set_description('Epoch {}, loss: {:.8f}, smooth loss: {:.8f}'.format(epoch + 1, loss.item(),
                                                                                          loss_dict['smooth'].item()))
            else:
                pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss.mean().item())

    mean_loss = np.stack(losses).mean()
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}'.format(epoch + 1, mean_loss))
    return mean_loss


def test_one_epoch(dataloader, epoch):
    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    model.eval()
    metrics.clear()
    losses = []
    running_time = []
    with tqdm(dataloader) as pbar:
        for data_dict in pbar:
            data_dict = to_device(data_dict, device)
            with torch.no_grad():
                time_start = perf_counter()
                # Use 'tir8' key consistent with train_transpose.py
                res = model(data_dict['rgb'], data_dict['depth'], data_dict['tir8'])
                time_end = perf_counter()
                n, h, w = data_dict['depth'].shape
                data_dict['pred'] = res.view(n, h, w)
                loss_dict = criterion(data_dict)
                loss = loss_dict['loss']
                _ = metrics.evaluate_batch(data_dict, record=True)

            duration = time_end - time_start
            if 'smooth' in loss_dict.keys():
                pbar.set_description('Epoch {}, loss: {:.8f}, smooth loss: {:.8f}'.format(epoch + 1, loss.item(),
                                                                                          loss_dict['smooth'].item()))
            else:
                pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss.item())
            running_time.append(duration)

    mean_loss = np.stack(losses).mean()
    avg_running_time = np.stack(running_time).mean()
    logger.info(
        'Finish testing process in epoch {}, mean testing loss: {:.8f}, average running time: {:.4f}s'.format(epoch + 1,
                                                                                                              mean_loss,
                                                                                                              avg_running_time))
    metrics_result = metrics.get_results()
    metrics.display_results()
    return mean_loss, metrics_result


def train(start_epoch):
    if start_epoch != 0:
        min_loss = checkpoint_loss
        min_loss_epoch = start_epoch
        display_results(checkpoint_metrics, logger)
    else:
        min_loss = LOSS_INF
        min_loss_epoch = None

    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_loss = train_one_epoch(epoch)
        val_loss, val_metrics = test_one_epoch(test_dataloader, epoch)

        # Optional: test on real data (commented out same as train_transpose.py)
        # test_real_loss, _ = test_one_epoch(test_real_dataloader, epoch)

        tensorboard_log.add_scalar("train_mean_loss", train_loss, epoch)
        tensorboard_log.add_scalar("real_mean_loss", val_loss, epoch)
        # tensorboard_log.add_scalar("syn_mean_loss", test_real_loss, epoch)

        criterion.step()
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if builder.multigpu() else model.state_dict(),
            'loss': val_loss,
            'metrics': val_metrics
        }
        torch.save(save_dict, os.path.join(stats_dir, 'checkpoint-epoch{}.tar'.format(epoch)))
        if val_loss < min_loss:
            min_loss = val_loss
            min_loss_epoch = epoch + 1
            torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'))

    logger.info('Fine-tuning Finished. Min testing loss: {:.6f}, in epoch {}'.format(min_loss, min_loss_epoch))
    tensorboard_log.close()


if __name__ == '__main__':
    train(start_epoch=start_epoch)

    