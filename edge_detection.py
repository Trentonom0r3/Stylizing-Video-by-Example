import torch
from PIL import Image
import numpy as np
import os
from phycv import PST_GPU, PAGE_GPU
import cv2

def compute_edge(img_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the image
    original_image = Image.open(img_file).convert("L")  # Convert image to grayscale

    # Save the new grayscale image
    grayscale_image_file = img_file.rsplit(".", 1)[0] + "_grayscale.png"
    original_image.save(grayscale_image_file)

    # Check if edges directory exists, if not create it
    edge_output_dir = os.path.join(os.path.dirname(img_file), "edges")
    os.makedirs(edge_output_dir, exist_ok=True)

    # Apply edge detection
    edge_file = os.path.join(edge_output_dir, os.path.basename(grayscale_image_file))

    # PST parameters
    S = 0.3
    W = 15
    sigma_LPF = 0.15
    thresh_min = 0.05
    thresh_max = 0.9
    morph_flag = 1

    pst_gpu = PST_GPU(device=device)
    edge_map = pst_gpu.run(
        grayscale_image_file, S, W, sigma_LPF, thresh_min, thresh_max, morph_flag
    )
    edge_map = edge_map.cpu().numpy()
    # Apply Gaussian blur
    edge_map = cv2.GaussianBlur(edge_map, (0, 0), 6)  # You can adjust the kernel size (5, 5) and sigma (0) as needed

    edge_result = Image.fromarray((edge_map * 255).astype(np.uint8))
    edge_result.save(edge_file)

    return edge_file

def create_gaussian_kernel(size, sigma):
    """Create a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * 
                     np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)), 
        (size, size))
    return kernel / np.sum(kernel)

def compute_edge_guide(img_file):
    # Load the image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    
    # Ensure the image is in BGR format
    if len(img.shape) == 2:
        # If the image is grayscale, convert it to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        # If the image has an alpha channel, remove it
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur using custom Gaussian kernel
    size = 5  # the size of the kernel in the C++ code
    sigma = 6.0  # the sigma value in the C++ code
    kernel = create_gaussian_kernel(size, sigma)
    blurred = cv2.filter2D(gray, -1, kernel)
    
    # Subtract the blurred image from the grayscale image
    edge_guide = cv2.subtract(gray, blurred)

    # Add constant to all pixel values
    edge_guide = cv2.add(edge_guide, 0.5 * 255)  # Multiply by 255 because OpenCV uses 0-255 range for pixel values
    
    # Ensure the values are within the range [0, 255]
    edge_guide = np.clip(edge_guide, 0, 255)

    # Convert back to BGR for saving (all channels will be the same)
    edge_guide_bgr = cv2.cvtColor(edge_guide.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Save and return the edge guide
    edge_guide_file = img_file.replace('.png', '_edge_guide.png')
    cv2.imwrite(edge_guide_file, edge_guide_bgr)
    
    return edge_guide_file

def compute_edge_PAGE(img_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the image
    original_image = Image.open(img_file).convert("L")  # convert to grayscale

    # Save the new grayscale image
    grayscale_image_file = img_file.rsplit(".", 1)[0] + "_grayscale.png"
    original_image.save(grayscale_image_file)

    # Check if edges directory exists, if not create it
    edge_output_dir = os.path.join(os.path.dirname(img_file), "edges")
    os.makedirs(edge_output_dir, exist_ok=True)

    # Apply edge detection
    edge_file = os.path.join(edge_output_dir, os.path.basename(grayscale_image_file))
    # PAGE parameters
    mu_1 = 0
    mu_2 = 0.35
    sigma_1 = 0.05
    sigma_2 = 0.8
    S1 = 0.8
    S2 = 0.8
    sigma_LPF = 0.1
    thresh_min = 0.0
    thresh_max = 0.9
    morph_flag = 1

    page_gpu = PAGE_GPU(direction_bins=10, device=device)
    edge_map = page_gpu.run(
        grayscale_image_file,
        mu_1,
        mu_2,
        sigma_1,
        sigma_2,
        S1,
        S2,
        sigma_LPF,
        thresh_min,
        thresh_max,
        morph_flag,
    )
    edge_map = edge_map.cpu().numpy()

    # If edge_map is a color image, convert it to grayscale
    if edge_map.ndim == 3:
        edge_map = cv2.cvtColor(edge_map, cv2.COLOR_RGB2GRAY)

    # Convert original image to numpy array
    original_array = np.array(original_image) / 255.0

    # Subtract PAGE_GPU result from original image
    edge_map = original_array - edge_map

    # Apply Gaussian blur
    edge_map = cv2.GaussianBlur(edge_map, (5, 5), 6)  # You can adjust the kernel size (5, 5) and sigma (6) as needed

    # Add constant to all pixel values
    edge_map = edge_map + 0.5

    # Ensure the values are within the range [0, 1]
    edge_map = np.clip(edge_map, 0, 1)

    edge_result = Image.fromarray((edge_map * 255).astype(np.uint8))
    
    edge_result.save(edge_file)

    return edge_file
