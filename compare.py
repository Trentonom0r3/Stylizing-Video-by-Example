import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, warped_gradient):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo, warped_gradient], axis=0)

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # Create initial, unwarped gradient
            gradient = create_gradient(image1.shape)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            # Warp the gradient based on the optical flow
            warped_gradient = warp_gradient(gradient, flow_up)
            
            viz(image1, flow_up, warped_gradient)


def create_gradient(shape):
    _, _, h, w = shape
    gradient = np.zeros((1, 3, h, w))

    for i in range(h):
        for j in range(w):
            gradient[0, 0, i, j] = j / w  # Red channel
            gradient[0, 1, i, j] = i / h  # Green channel

    return torch.tensor(gradient).float().to(DEVICE)



def warp_gradient(gradient, flow):
    _, _, h, w = flow.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid = grid.to(DEVICE) + flow[0]

    # normalize grid to [-1,1] to match the input of F.grid_sample
    grid[:, :, 0] = 2.0 * grid[:, :, 0].clone() / max(w - 1, 1) - 1.0
    grid[:, :, 1] = 2.0 * grid[:, :, 1].clone() / max(h - 1, 1) - 1.0

    warped_gradient = torch.nn.functional.grid_sample(gradient, grid.permute(0, 2, 3, 1))

    return warped_gradient



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
