"""
Improved Depth Inference Script
Modified to match the processing pipeline of sample_inference.py
For cases where we don't have a separate mask file
"""
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from inference import Inferencer
import matplotlib.pyplot as plt
import os
import glob
import yaml
import argparse
from tqdm import tqdm
from utils.data_preparation import process_data, exr_loader


DILATION_KERNEL = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]]).astype(np.uint8)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def dropout_random_ellipses_4corruptmask(mask, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
    """
    dropout_mask = mask.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    zero_pixel_indices = np.array(np.where(dropout_mask == 0)).T
    if zero_pixel_indices.shape[0] == 0:
        return dropout_mask

    dropout_centers_indices = np.random.choice(zero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = zero_pixel_indices[dropout_centers_indices, :]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        tmp_mask = np.zeros_like(dropout_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        dropout_mask[tmp_mask == 1] = 1

    return dropout_mask


def handle_depth(depth, depth_gt, depth_gt_mask):
    """
    Handle depth image with inpainting and random ellipse augmentation
    Exactly as in sample_inference.py
    """
    depth[depth_gt_mask==1] = 0
    depth_gt_mask_uint8 = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
    depth_gt_mask_uint8[depth_gt_mask_uint8 != 0] = 1

    if depth.max() <= 0:
        return depth

    depth_uint8 = depth.copy() / depth.max() * 255
    depth_uint8 = np.array(depth_uint8, dtype=np.uint8)
    depth_uint8 = cv2.inpaint(depth_uint8, depth_gt_mask_uint8, 5, cv2.INPAINT_NS)
    depth_uint8 = np.array(depth_uint8, dtype=np.float32) / 255 * depth.max()

    mask_pixel_indices1 = np.array(np.where(depth_gt_mask == 1)).T
    if mask_pixel_indices1.shape[0] == 0:
        return depth

    dropout_size = int(mask_pixel_indices1.shape[0] * 0.003)
    if dropout_size == 0:
        return depth

    dropout_centers_indices = np.random.choice(mask_pixel_indices1.shape[0], size=dropout_size)
    dropout_centers = mask_pixel_indices1[dropout_centers_indices, :]
    x_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    y_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    angles = np.random.randint(0, 360, size=dropout_size)

    result_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)

    for i in range(dropout_size // 2):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        tmp_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius),
                              angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        result_mask[tmp_mask == 1] = 1

    mask = np.logical_and(result_mask, depth_gt_mask_uint8)
    depth[mask==1] = depth_uint8[mask == 1]

    result_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)

    for i in range(dropout_size - dropout_size // 2):
        center = dropout_centers[i + dropout_size // 2, :]
        x_radius = np.round(x_radii[i + dropout_size // 2]).astype(int)
        y_radius = np.round(y_radii[i + dropout_size // 2]).astype(int)
        angle = angles[i + dropout_size // 2]

        tmp_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius),
                              angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        result_mask[tmp_mask == 1] = 1

    mask = np.logical_and(result_mask, depth_gt_mask_uint8)
    depth[mask==1] = depth_gt[mask==1]

    return depth


def draw_point_cloud(color, depth, camera_intrinsics, use_mask=False, use_inpainting=True,
                    scale=1000.0, inpainting_radius=5, fault_depth_limit=0.2, epsilon=0.01):
    """Generate point cloud from depth image"""
    d = depth.copy()
    c = color.copy() / 255.0

    if use_inpainting:
        fault_mask = (d < fault_depth_limit * scale)
        d[fault_mask] = 0
        inpainting_mask = (np.abs(d) < epsilon * scale).astype(np.uint8)
        d = cv2.inpaint(d, inpainting_mask, inpainting_radius, cv2.INPAINT_NS)

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = d / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)

    if use_mask:
        mask = (points_z > 0)
        points = points[mask]
        c = c[mask]
    else:
        points = points.reshape((-1, 3))
        c = c.reshape((-1, 3))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(c)
    return cloud


def process_single_image(inferencer, rgb_path, depth_path, output_path, config,
                        mask_path=None, show_plot=False, generate_pointcloud=False,
                        camera_intrinsics=None, mode='batch'):
    """
    Process single image - exactly matching sample_inference.py processing flow
    """
    try:
        inference_config = config['inference']
        processing_config = config['processing']
        vis_config = config['visualization']

        target_size = tuple(inference_config['target_size'])
        depth_scale = inference_config['depth_scale']
        depth_range = tuple(inference_config['depth_range'])

        # Load images - exactly as in sample_inference.py
        rgb = np.array(Image.open(rgb_path), dtype=np.float32)
        depth = np.array(Image.open(depth_path), dtype=np.float32)

        # Convert depth to meters
        depth = depth / depth_scale

        # Create depth_gt from input depth (since we don't have separate GT)
        depth_gt = depth.copy()

        # Save original copies
        rgbcopy = rgb.copy()
        depth_copy = depth.copy()

        # Run inference BEFORE processing masks - critical difference!
        res = inferencer.inference(rgb, depth)

        # Process NaN values
        depth_gt[np.isnan(depth_gt)] = 0.0

        # Generate mask from depth_gt AFTER NaN processing but BEFORE resizing
        rgb_mask = np.where(depth_gt < 0.000000001, 255, 0).astype(np.uint8)

        # Resize all images to target size
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        depth_copy = cv2.resize(depth_copy, target_size, interpolation=cv2.INTER_NEAREST)
        depth_gt = cv2.resize(depth_gt, target_size, interpolation=cv2.INTER_NEAREST)
        res = cv2.resize(res, target_size, interpolation=cv2.INTER_NEAREST)
        rgb_mask = cv2.resize(rgb_mask, target_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Process mask - convert to binary
        rgb_mask[rgb_mask != 0] = 1

        # Create depth mask
        depth_mask = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
        depth_mask[depth_mask != 0] = 1

        # Apply handle_depth - critical processing step
        depth_gt_new = handle_depth(depth_gt.copy(), depth_gt.copy(), rgb_mask)

        # Clean up depth values - apply depth range limits
        neg_zero_mask = np.where(depth_gt_new < 0.0000001)
        res[neg_zero_mask] = 0
        depth_gt_new[neg_zero_mask] = 0
        depth[neg_zero_mask] = 0

        neg_zero_mask = np.where(depth_gt_new > depth_range[1])
        res[neg_zero_mask] = 0
        depth_gt_new[neg_zero_mask] = 0
        depth[neg_zero_mask] = 0

        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save result
        if processing_config['save_16bit']:
            result_depth = (res * depth_scale).astype(np.uint16)
            cv2.imwrite(output_path, result_depth)
            print(f"Saved 16-bit result to: {output_path}")
        else:
            np.save(output_path.replace('.png', '.npy'), res)
            print(f"Saved numpy result to: {output_path.replace('.png', '.npy')}")

        # Create visualization matching sample_inference.py EXACTLY
        fig, axs = plt.subplots(2, 2, figsize=tuple(vis_config['figsize']))
        tt = vis_config.get('colormap', 'hsv')

        # RGB - use original copy
        rgb_1 = rgbcopy.astype(np.uint8)  # Changed from int8 to uint8
        axs.flat[0].imshow(rgb_1)
        axs.flat[0].set_title("rgb")
        axs.flat[0].axis('off')

        # Original depth mask
        axs.flat[1].imshow(rgb_mask, cmap=tt)
        axs.flat[1].set_title("original")
        axs.flat[1].axis('off')

        # Model output
        axs.flat[2].imshow(res, cmap=tt)
        axs.flat[2].set_title("model output")
        axs.flat[2].axis('off')

        # Ground truth (processed depth)
        axs.flat[3].imshow(depth_gt_new, cmap=tt)
        axs.flat[3].set_title("ground truth")
        axs.flat[3].axis('off')

        plt.tight_layout()

        if show_plot:
            plt.show()

        # Save visualization
        if processing_config['save_visualization']:
            if mode == 'single_test':
                visualization_dir = os.path.join(output_dir, 'visualizations')
            else:
                visualization_dir = config['paths'].get('visualization_dir',
                                                     os.path.join(output_dir, 'visualizations'))

            if not os.path.exists(visualization_dir):
                os.makedirs(visualization_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(output_path))[0]
            viz_path = os.path.join(visualization_dir, f'{base_name}_comparison.{vis_config["save_format"]}')
            plt.savefig(viz_path, dpi=vis_config['dpi'], bbox_inches='tight')
            print(f"Saved visualization to: {viz_path}")

        plt.close()

        # Generate point cloud if requested
        if generate_pointcloud and camera_intrinsics is not None:
            try:
                rgb_resized = cv2.resize(rgbcopy, target_size, interpolation=cv2.INTER_NEAREST)
                cloud = draw_point_cloud(rgb_resized, res, camera_intrinsics, scale=1.0)
                cloud_gt = draw_point_cloud(rgb_resized, depth_gt_new, camera_intrinsics, scale=1.0)

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                o3d.visualization.draw_geometries([cloud, cloud_gt, frame, sphere])
            except Exception as e:
                print(f"Point cloud visualization failed: {str(e)}")

        # Return results
        results = {
            'res': res,
            'depth': depth,
            'depth_gt_new': depth_gt_new,
            'rgb_mask': rgb_mask,
            'rgbcopy': rgbcopy
        }

        return True, f"Successfully processed {os.path.basename(rgb_path)}", results

    except Exception as e:
        return False, f"Error processing {os.path.basename(rgb_path)}: {str(e)}", None

def process_single_test_image(inferencer, rgb_path, depth_path, depth_gt_path, output_path, config,
                        mask_path=None, show_plot=False, generate_pointcloud=False,
                        camera_intrinsics=None, mode='batch'):
    """
    Process single image - exactly matching sample_inference.py processing flow
    """
    try:
        inference_config = config['inference']
        processing_config = config['processing']
        vis_config = config['visualization']

        target_size = tuple(inference_config['target_size'])
        depth_scale = inference_config['depth_scale']
        depth_range = tuple(inference_config['depth_range'])

        # Load images - exactly as in sample_inference.py
        rgb = np.array(Image.open(rgb_path), dtype=np.float32)
        depth = np.array(Image.open(depth_path), dtype=np.float32)

        # Convert depth to meters
        depth = depth / depth_scale

        # Create depth_gt from input depth (since we don't have separate GT)
        depth_gt = np.array(Image.open(depth_gt_path), dtype=np.float32)
        depth_gt = depth_gt / depth_scale

        # Save original copies
        rgbcopy = rgb.copy()
        depth_copy = depth.copy()

        # Run inference BEFORE processing masks - critical difference!
        res = inferencer.inference(rgb, depth)

        # Process NaN values
        depth_gt[np.isnan(depth_gt)] = 0.0

        # Generate mask from depth_gt AFTER NaN processing but BEFORE resizing
        rgb_mask = np.where(depth_gt < 0.000000001, 255, 0).astype(np.uint8)

        # Resize all images to target size
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        depth_copy = cv2.resize(depth_copy, target_size, interpolation=cv2.INTER_NEAREST)
        depth_gt = cv2.resize(depth_gt, target_size, interpolation=cv2.INTER_NEAREST)
        res = cv2.resize(res, target_size, interpolation=cv2.INTER_NEAREST)
        rgb_mask = cv2.resize(rgb_mask, target_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Process mask - convert to binary
        rgb_mask[rgb_mask != 0] = 1

        # Create depth mask
        depth_mask = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
        depth_mask[depth_mask != 0] = 1

        # Apply handle_depth - critical processing step
        depth_gt_new = handle_depth(depth_gt.copy(), depth_gt.copy(), rgb_mask)

        # Clean up depth values - apply depth range limits
        neg_zero_mask = np.where(depth_gt_new < 0.0000001)
        res[neg_zero_mask] = 0
        depth_gt_new[neg_zero_mask] = 0
        depth[neg_zero_mask] = 0

        neg_zero_mask = np.where(depth_gt_new > depth_range[1])
        res[neg_zero_mask] = 0
        depth_gt_new[neg_zero_mask] = 0
        depth[neg_zero_mask] = 0

        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save result
        if processing_config['save_16bit']:
            result_depth = (res * depth_scale).astype(np.uint16)
            cv2.imwrite(output_path, result_depth)
            print(f"Saved 16-bit result to: {output_path}")
        else:
            np.save(output_path.replace('.png', '.npy'), depth)
            print(f"Saved numpy result to: {output_path.replace('.png', '.npy')}")

        # Create visualization matching sample_inference.py EXACTLY
        fig, axs = plt.subplots(2, 2, figsize=tuple(vis_config['figsize']))
        tt = vis_config.get('colormap', 'hsv')

        # RGB - use original copy
        rgb_1 = rgbcopy.astype(np.uint8)  # Changed from int8 to uint8
        axs.flat[0].imshow(rgb_1)
        axs.flat[0].set_title("rgb")
        axs.flat[0].axis('off')

        # Original depth mask
        axs.flat[1].imshow(rgb_mask, cmap=tt)
        axs.flat[1].set_title("original")
        axs.flat[1].axis('off')

        # Model output
        axs.flat[2].imshow(depth, cmap=tt)
        axs.flat[2].set_title("model output")
        axs.flat[2].axis('off')

        # Ground truth (processed depth)
        axs.flat[3].imshow(depth_gt, cmap=tt)
        axs.flat[3].set_title("ground truth")
        axs.flat[3].axis('off')

        plt.tight_layout()

        if show_plot:
            plt.show()

        # Save visualization
        if processing_config['save_visualization']:
            if mode == 'single_test':
                visualization_dir = os.path.join(output_dir, 'visualizations')
            else:
                visualization_dir = config['paths'].get('visualization_dir',
                                                     os.path.join(output_dir, 'visualizations'))

            if not os.path.exists(visualization_dir):
                os.makedirs(visualization_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(output_path))[0]
            viz_path = os.path.join(visualization_dir, f'{base_name}_comparison.{vis_config["save_format"]}')
            plt.savefig(viz_path, dpi=vis_config['dpi'], bbox_inches='tight')
            print(f"Saved visualization to: {viz_path}")

        plt.close()

        # Generate point cloud if requested
        if generate_pointcloud and camera_intrinsics is not None:
            try:
                rgb_resized = cv2.resize(rgbcopy, target_size, interpolation=cv2.INTER_NEAREST)
                cloud = draw_point_cloud(rgb_resized, res, camera_intrinsics, scale=1.0)
                cloud_gt = draw_point_cloud(rgb_resized, depth_gt_new, camera_intrinsics, scale=1.0)

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
                o3d.visualization.draw_geometries([cloud, cloud_gt, frame, sphere])
            except Exception as e:
                print(f"Point cloud visualization failed: {str(e)}")

        # Return results
        results = {
            'res': res,
            'depth': depth,
            'depth_gt_new': depth_gt_new,
            'rgb_mask': rgb_mask,
            'rgbcopy': rgbcopy
        }

        return True, f"Successfully processed {os.path.basename(rgb_path)}", results

    except Exception as e:
        return False, f"Error processing {os.path.basename(rgb_path)}: {str(e)}", None

def batch_depth_completion(config):
    """Batch process depth completion"""
    print("Initializing inferencer...")
    inferencer = Inferencer()

    paths_config = config['paths']
    files_config = config['files']
    processing_config = config['processing']

    rgb_dir = paths_config['rgb_dir']
    depth_dir = paths_config['depth_dir']
    output_dir = paths_config['output_dir']
    mask_dir = paths_config.get('mask_dir', None)

    rgb_files = glob.glob(os.path.join(rgb_dir, files_config['rgb_pattern']))
    rgb_files.sort()

    if not rgb_files:
        print(f"No RGB images found in {rgb_dir} with pattern {files_config['rgb_pattern']}")
        return

    print(f"Found {len(rgb_files)} RGB images")

    successful_count = 0
    failed_count = 0

    for rgb_file in tqdm(rgb_files, desc="Processing images"):
        rgb_basename = os.path.basename(rgb_file)
        base_name = os.path.splitext(rgb_basename)[0]

        if files_config['rgb_suffix'] and base_name.endswith(files_config['rgb_suffix']):
            base_name = base_name[:-len(files_config['rgb_suffix'])]

        # Build depth file path
        depth_ext = os.path.splitext(files_config['depth_pattern'])[1]
        depth_filename = base_name + files_config['depth_suffix'] + depth_ext
        depth_file = os.path.join(depth_dir, depth_filename)

        if not os.path.exists(depth_file):
            if processing_config['verbose']:
                print(f"Warning: Depth image not found for {rgb_basename}: {depth_filename}")
            failed_count += 1
            continue

        # Build mask file path if mask directory provided
        mask_file = None
        if mask_dir and 'mask_pattern' in files_config:
            mask_ext = os.path.splitext(files_config['mask_pattern'])[1]
            mask_filename = base_name + files_config.get('mask_suffix', '') + mask_ext
            mask_file = os.path.join(mask_dir, mask_filename)
            if not os.path.exists(mask_file):
                mask_file = None

        # Build output path
        output_filename = base_name + files_config['output_suffix'] + files_config['output_ext']
        output_file = os.path.join(output_dir, output_filename)

        success, message, _ = process_single_image(inferencer, rgb_file, depth_file,
                                                   output_file, config, mask_path=mask_file,
                                                   mode='batch')

        if success:
            successful_count += 1
            if processing_config['verbose']:
                print(message)
        else:
            failed_count += 1
            print(message)

    print(f"\nBatch processing completed:")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed: {failed_count} images")


def single_image_inference(config):
    """Single image inference"""
    print("Initializing inferencer...")
    inferencer = Inferencer()

    single_config = config['single_image']
    rgb_path = single_config['rgb_path']
    depth_path = single_config['depth_path']
    output_path = single_config['output_path']
    mask_path = single_config.get('mask_path', None)

    success, message, _ = process_single_test_image(inferencer, rgb_path, depth_path,
                                               output_path, config, mask_path=mask_path,
                                               mode='single')
    print(message)

    return success


def single_test_with_visualization(config):
    """
    Single test with full visualization - exactly matching sample_inference.py
    """
    print("Running single test with full visualization...")

    # Initialize inferencer
    inferencer = Inferencer()

    # Get paths from config
    test_config = config['single_test']
    rgb_path = test_config['rgb_path']
    depth_path = test_config['depth_path']
    depth_gt_path = test_config['depth_gt_path']
    output_path = test_config['output_path']
    mask_path = test_config.get('mask_path', None)

    # Validate input files exist
    if not os.path.exists(rgb_path):
        return False, f"RGB file not found: {rgb_path}"
    if not os.path.exists(depth_path):
        return False, f"Depth file not found: {depth_path}"

    print(f"Processing RGB: {rgb_path}")
    print(f"Processing Depth: {depth_path}")
    if mask_path:
        print(f"Using mask: {mask_path}")

    # Load camera intrinsics if available
    camera_intrinsics = None
    camera_intrinsics_path = test_config.get('camera_intrinsics_path')
    if camera_intrinsics_path and os.path.exists(camera_intrinsics_path):
        camera_intrinsics = np.load(camera_intrinsics_path)
        print(f"Loaded camera intrinsics from: {camera_intrinsics_path}")

        # Scale camera intrinsics to match target size
        target_size = tuple(config['inference']['target_size'])
        original_size = (640, 480)  # Assuming D435 default resolution
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]

        camera_intrinsics_scaled = camera_intrinsics.copy()
        camera_intrinsics_scaled[0, 0] *= scale_x  # fx
        camera_intrinsics_scaled[1, 1] *= scale_y  # fy
        camera_intrinsics_scaled[0, 2] *= scale_x  # cx
        camera_intrinsics_scaled[1, 2] *= scale_y  # cy

        print(f"Scaled camera intrinsics from {original_size} to {target_size}")
        camera_intrinsics = camera_intrinsics_scaled

    # Process image with visualization
    success, message, results = process_single_test_image(
        inferencer, rgb_path, depth_path, depth_gt_path, output_path, config,
        mask_path=mask_path,
        show_plot=True,
        generate_pointcloud=config['processing'].get('generate_pointcloud', False),
        camera_intrinsics=camera_intrinsics,
        mode='single_test'
    )

    if success and results:
        # Print analysis
        res = results['res']
        depth = results['depth']
        depth_gt_new = results['depth_gt_new']

        print("\n=== Analysis Results ===")
        print(f"Valid pixels - Original: {np.sum(depth > 0)}")
        print(f"Valid pixels - Completed: {np.sum(res > 0)}")
        print(f"Valid pixels - Ground truth: {np.sum(depth_gt_new > 0)}")

        if np.sum(res > 0) > 0:
            print(f"Depth range - Completed: [{np.min(res[res > 0]):.3f}, {np.max(res):.3f}]")

        print(f"Mean absolute difference (res vs depth): {np.mean(np.abs(res - depth)):.3f}")
        print(f"Mean absolute difference (res vs GT): {np.mean(np.abs(res - depth_gt_new)):.3f}")

        # Print file locations
        print(f"\n=== Output Files ===")
        print(f"Result saved to: {output_path}")

        output_dir = os.path.dirname(output_path)
        visualization_dir = os.path.join(output_dir, 'visualizations')
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        viz_path = os.path.join(visualization_dir, f'{base_name}_comparison.png')
        if os.path.exists(viz_path):
            print(f"Visualization saved to: {viz_path}")

    return success, message


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Depth Completion Inference')
    parser.add_argument('mode', choices=['single', 'batch', 'single_test'],
                       help='Processing mode: single image, batch processing, or single test')
    parser.add_argument('--config', type=str, default='configs/depth_inference.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        return

    config = load_config(args.config)

    if args.mode == 'single':
        print("Running single image inference...")
        success = single_image_inference(config)
        if success:
            print("Single image processing completed successfully!")
        else:
            print("Single image processing failed!")

    elif args.mode == 'batch':
        print("Running batch inference...")
        batch_depth_completion(config)

    elif args.mode == 'single_test':
        print("Running single test with full analysis...")
        success, message = single_test_with_visualization(config)
        print(message)
        if success:
            print("Single test completed successfully!")
        else:
            print("Single test failed!")

    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()