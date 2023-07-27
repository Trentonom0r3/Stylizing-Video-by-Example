import sys
import os
sys.path.append('core')
import torch
from PIL import Image
import numpy as np
from utils.utils import InputPadder
from utils import flow_viz
import cv2

DEVICE = 'cuda'

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
    

    # Define flow here
    flow = cv2.resize(flow_up[0].permute(1,2,0).cpu().numpy(), original_size)

    # Convert the flow visualization to an image
    flow_viz_img = flow_viz.flow_to_image(flow)

    # Resize the flow visualization image back to the original size
    flow_viz_img_resized = cv2.resize(flow_viz_img, original_size)

    # Save the resized flow visualization
    d_file = os.path.join(flow_output_dir, f'input{i+1}.png')
    Image.fromarray((flow_viz_img_resized * 255).astype(np.uint8)).save(d_file)
    
    # Save the raw flow vectors
    d_file_raw = d_file.replace('.png', '.npy')
    np.save(d_file_raw, flow_up[0].permute(1,2,0).cpu().numpy())

     # Create a coordinate map and encode it into an image only if it's the first iteration
    if coord_map is None:
        h, w = original_size[::-1]
        coord_map = np.zeros((h, w, 3), dtype=np.uint8)
        coord_map[..., 0] = np.linspace(0, 255, w)  # x-coordinate in the red channel
        coord_map[..., 1] = np.linspace(0, 255, h)[:, np.newaxis]  # y-coordinate in the green channel
        coord_map_warped = coord_map.copy()

    # Adjust the flow values to absolute positions
    h, w = flow.shape[:2]
    flow[..., 0] += np.arange(w)
    flow[..., 1] += np.arange(h)[:, np.newaxis]

    # Manually implement the warping
    coord_map_warped = backward_mapping(coord_map_warped, flow)

    # Save the warped coordinate map as the G_pos guide
    g_pos_file = d_file.replace('.png', '_g_pos.png')
    Image.fromarray(coord_map_warped).save(g_pos_file)
    
    return d_file, d_file_raw, g_pos_file


# Function to warp guide images into the next frame using optical flow
def warp_image(image, flow):
    h, w = image.shape[:2]
    if flow is not None:
        flow = cv2.resize(flow, (w, h)) # Resize the flow to match the image size
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        warped_image = backward_mapping(image, flow)
    else:
        warped_image = image
    return warped_image


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