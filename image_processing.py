import torch
import cv2
import numpy as np
from PIL import Image

DEVICE = 'cuda'

def load_image(imfile):
    img = Image.open(imfile).convert("RGB")
    size = img.size
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.to(DEVICE), size

def warp_image(stylized_image, flow):
    h, w = stylized_image.shape[:2]
    flow = cv2.resize(flow, (w, h)) # Resize the flow to match the image size
    flow[:,:,0] += np.arange(w) # Add indices to x displacements
    flow[:,:,1] += np.arange(h)[:,np.newaxis] # Add indices to y displacements
    warped_image = cv2.remap(stylized_image, flow, None, cv2.INTER_LINEAR)
    return warped_image

def warp_guide(guide, flow):
    h, w = guide.shape[:2]
    flow = cv2.resize(flow, (w, h))
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    warped_guide = cv2.remap(guide, flow, None, cv2.INTER_LINEAR)
    return warped_guide
