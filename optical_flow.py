import sys
import os
from matplotlib import pyplot as plt
sys.path.append('core')
import torch
from PIL import Image
import numpy as np
from utils.utils import InputPadder
from utils import flow_viz
import cv2
from PIL import Image, ImageDraw, ImageFilter
import torch.nn.functional as F

DEVICE = 'cuda'

def warp(x, flo):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    print(f"Grid size: {grid.size()}")
    print(f"Flow size: {flo.size()}")

    if x.is_cuda:
        grid = grid.cuda()

    # Resize flo to match the size of grid
    flo = F.interpolate(flo, size=(H, W), mode='bilinear', align_corners=False)
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).to(DEVICE)
    mask = F.grid_sample(mask, vgrid)
    print(f"vgrid size: {vgrid.size()}")

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1
    print(f"vgrid size: {vgrid.size()}")

    return output
# This function computes the optical flow between two images using the RAFT model,
# generates a visualization of the flow, saves the visualization and raw flow vectors,
# and uses the optical flow to warp a coordinate map, which is then saved as well.
# Inputs:
# - model: the RAFT model
# - image1, image2: the two images between which to compute the optical flow
# - flow_output_dir: the directory where to save the flow visualization and raw vectors
# - i: the index of the image pair (used for naming the output files)
# - original_size: the original size of the images
# Outputs: paths to the saved flow visualization, raw flow vectors, and warped coordinate map


coord_map = None
coord_map_warped = None

def compute_optical_flow(model, image1, image2, flow_output_dir, i, original_size):
    global coord_map, coord_map_warped

    with torch.no_grad():
        images = [image1, image2]
        padder = InputPadder(images[0].shape)
        images = [padder.pad(im)[0] for im in images]
        flow_low, flow_up = model(images[0], images[1], iters=20, test_mode=True)
    
    print(flow_up.min(), flow_up.max())

    # Convert the optical flow to an image and a NumPy array
    flow = cv2.resize(flow_up[0].permute(1,2,0).cpu().numpy(), original_size)
    flow_viz_img = flow_viz.flow_to_image(flow)

    # Save the flow visualization
    d_file = os.path.join(flow_output_dir, f'input{i+1}.png')
    Image.fromarray((flow_viz_img * 255).astype(np.uint8)).save(d_file)
    
    # Save the raw flow vectors
    d_file_raw = d_file.replace('.png', '.npy')
    np.save(d_file_raw, flow_up[0].permute(1,2,0).cpu().numpy())

     # Create a coordinate map only if it's the first iteration
    if coord_map is None:
        h, w = original_size[::-1]
        coord_map = torch.zeros((3, h, w)).to(DEVICE)
        coord_map[0] = torch.linspace(0, 1, w)  # x-coordinate in the red channel
        coord_map[1] = torch.linspace(0, 1, h)[:, np.newaxis]  # y-coordinate in the green channel
        coord_map_warped = coord_map.clone()

    # Warp the coordinate map according to the optical flow
    coord_map_warped = warp(coord_map_warped.unsqueeze(0), flow_up).squeeze(0)

    # Save the warped coordinate map as the G_pos guide
    g_pos_file = d_file.replace('.png', '_g_pos.png')
    Image.fromarray((coord_map_warped.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(g_pos_file)
    
    return d_file, d_file_raw, g_pos_file

# Function to manually implement backward mapping and bilinear interpolation
def backward_mapping(src, flow):
    h, w = src.shape[:2]
    x = flow[..., 0]
    y = flow[..., 1]
    x_floor = np.floor(x).astype(int)
    y_floor = np.floor(y).astype(int)
    x_weight = x - x_floor
    y_weight = y - y_floor

    x_floor = np.clip(x_floor, 0, w-2)
    y_floor = np.clip(y_floor, 0, h-2)

    # Extend the dimensions of x_weight and y_weight
    x_weight = x_weight[..., np.newaxis]
    y_weight = y_weight[..., np.newaxis]

    output = (1-x_weight) * (1-y_weight) * src[y_floor, x_floor] \
            + x_weight * (1-y_weight) * src[y_floor, x_floor+1] \
            + (1-x_weight) * y_weight * src[y_floor+1, x_floor] \
            + x_weight * y_weight * src[y_floor+1, x_floor+1]
    
    return output.astype(np.uint8)