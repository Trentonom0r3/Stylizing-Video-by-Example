from argparse import Namespace
import os
import subprocess
import sys
sys.path.append('core')
from raft import RAFT
from utils import frame_utils
from argparse import Namespace
from optical_flow import compute_optical_flow  # assuming compute_optical_flow is in optical_flow.py
import torch
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from stylization import stylize_frame
from optical_flow import compute_optical_flow, warp_image
from edge_detection import compute_edge_PAGE, compute_edge_guide, compute_edge

def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE), img.shape[1:]


DEVICE = 'cuda'
def main(style_file, input_dir, start_frame):
    output_dir = os.path.abspath(os.path.join(input_dir, "..", "Output", "stylized"))
    os.makedirs(output_dir, exist_ok=True)

    args = Namespace()
    args.model = 'models/raft-sintel.pth'
    args.small = False
    args.mixed_precision = False
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    image_files = sorted(
        [os.path.join(input_dir, file) for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(os.path.basename(x))[0])))
    )

    print('Computing edge maps for all images...')
    edge_files = [compute_edge(img_file) for img_file in tqdm(image_files)]

    flow_output_dir = os.path.abspath(os.path.join(input_dir, "..", "Output", "flow"))
    os.makedirs(flow_output_dir, exist_ok=True)

    print('Computing optical flow for all images...')
    d_files = []
    for i in range(start_frame - 1, len(image_files)):  # Notice change here
        image1, size1 = load_image(image_files[i])
        if i < len(image_files) - 1:
            image2, size2 = load_image(image_files[i + 1])
        else:
            image2 = image1  # Use the last image itself if it's the last iteration
        assert size1 == size2, "The two images do not have the same size."
        d_file, d_file_raw, g_pos_file= compute_optical_flow(model, image1, image2, flow_output_dir, i, size1)
        d_files.append((d_file, d_file_raw, g_pos_file))



    print('Setting stylized frame 1...')
    o_i_file = style_file  # The style image is the stylized first frame
    o_hat_i_file = None  # No previous frame exists for the first frame

    for i in range(start_frame, len(image_files)):
        new_o_i_file = os.path.join(output_dir, os.path.basename(image_files[i]).replace('input', 'stylized'))

        # Warp the stylized image
        stylized_image = cv2.imread(o_i_file)
        flow = np.load(d_files[i-1][1]) if i > start_frame else None  # Adjusted index
        warped_image = warp_image(stylized_image, flow)
        o_hat_i_file = os.path.join(output_dir, f'warped{i + 1}.png')
        cv2.imwrite(o_hat_i_file, warped_image)
        print(f"Using flow calculated from frame {i} to frame {i+1} to warp {o_i_file} into {o_hat_i_file}")
        
        if i < len(image_files) - 1:
            _ = stylize_frame(
                style_file,  # 
                image_files[0],  # Gcol initial
                image_files[i],  # Gcol target frame, the next frame in the sequence
                edge_files[0],  # Gedge intiial
                edge_files[i],  # Gedge target frame, the next frame in the sequence 
                d_files[0][2] if i > start_frame else None,  # Gpos initial (Adjusted index)
                d_files[i-1][2], # Gpos target frame, the next frame in the sequence (Adjusted index)
                new_o_i_file,  # Output file
                style_file,  # Gtemp: Use the previously styled frame
                o_hat_i_file,  # Gtemp: Use the previously styled frame, warped to the next frame in the sequence
            )

        # The stylized output from this step becomes the style frame for the next step
        o_i_file = new_o_i_file
        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <style_file> <input_dir>")
        sys.exit(1)

    style_file = sys.argv[1]
    input_dir = sys.argv[2]

    main(style_file, input_dir, start_frame=1)