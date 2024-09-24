# %%
# import libraries
import pandas as pd
import numpy as np
import ast
import cv2
import math
from typing import List
from PIL import Image
from pathlib import Path
from sklearn.metrics import average_precision_score
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.transforms import IndexSplitter
from scipy.ndimage.morphology import binary_erosion
from sklearn.metrics import f1_score
import os
import shutil
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import transforms

## function definitions
# preprocessing
def crop_files(image_dir, bbox_dir, mask_names, output_dir, file_suffix='_cropped'):
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg'):
            base_name = img_file.replace('.jpg', '')
            img_path = os.path.join(image_dir, img_file)
            bbox_file = os.path.join(bbox_dir, f"{base_name}_ROI_bbox.txt")

            with open(bbox_file, 'r') as f:
                bbox_data = ast.literal_eval(f.read())
                top, left, height, width = bbox_data.values()
                bottom = top + height
                right = left + width

            img = Image.open(img_path)
            img_cropped = img.crop((left, top, right, bottom))

            # Convert to RGB if it's RGBA
            if img_cropped.mode == 'RGBA':
                img_cropped = img_cropped.convert('RGB')

            img_cropped.save(os.path.join(output_dir, base_name + file_suffix + '.out.jpg'))

            # Crop corresponding mask files
            for mask_name in mask_names:
                mask_file = os.path.join(image_dir, f"{base_name}_{mask_name}_L.png")
                if os.path.exists(mask_file):
                    mask = Image.open(mask_file)
                    mask_cropped = mask.crop((left, top, right, bottom))
                    mask_cropped.save(os.path.join(output_dir, f"{base_name}_{mask_name}_L.png"))

def merge_masks(masks, output_path, base_image_size):
    combined_mask = np.zeros((base_image_size[1], base_image_size[0]), dtype=np.uint8)
    for mask_path in masks:
        current_mask = Image.open(mask_path).convert('L')
        if current_mask.size != base_image_size:
            current_mask = current_mask.resize(base_image_size, Image.NEAREST)
        combined_mask = np.maximum(combined_mask, np.array(current_mask))
    Image.fromarray(combined_mask).save(output_path)

def check_mask_dimensions(base_image_dir, mask_dir, mask_names):
    mismatches = []
    base_images = [f for f in os.listdir(base_image_dir) if f.endswith('.jpg')]

    for base_image_name in base_images:
        base_image_path = os.path.join(base_image_dir, base_image_name)
        base_image = Image.open(base_image_path)
        base_size = base_image.size  # (width, height)
        base_stem = os.path.splitext(base_image_name)[0].replace('_cropped.out', '')

        for mask_name in mask_names:
            mask_path = os.path.join(mask_dir, f"{base_stem}_{mask_name}_L.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                if mask.size != base_size:
                    mismatches.append((base_image_path, mask_path, base_size, mask.size))

    return mismatches

def correct_mask_dimensions(base_image_path, mask_dir, mask_names):
    """
    Correct the dimensions of masks to match their corresponding base images,
    including exact matching and inversion of axes if needed.

    Args:
    base_image_path (str): Path to the base image.
    mask_dir (str): Directory containing the mask files.
    mask_names (list): List of mask identifiers used in mask filenames.
    """
    base_image = Image.open(base_image_path)
    base_size = base_image.size  # (width, height)

    base_stem = os.path.splitext(os.path.basename(base_image_path))[0].replace('_cropped.out', '')

    for mask_name in mask_names:
        mask_path = os.path.join(mask_dir, f"{base_stem}_{mask_name}_L.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask_array = np.array(mask)

            # Check if mask dimensions need to be adjusted
            if mask_array.shape[1] != base_size[0] or mask_array.shape[0] != base_size[1]:  # (height, width)
                # Calculate the amount to crop from each side
                delta_w = mask_array.shape[1] - base_size[0]
                delta_h = mask_array.shape[0] - base_size[1]

                # Perform cropping to adjust width and height
                new_left = delta_w // 2
                new_upper = delta_h // 2
                new_right = mask_array.shape[1] - (delta_w - new_left)
                new_lower = mask_array.shape[0] - (delta_h - new_upper)

                mask_array = mask_array[new_upper:new_lower, new_left:new_right]

            # Save corrected mask
            Image.fromarray(mask_array).save(mask_path)
            print(f"Corrected dimensions for {mask_name} to {mask_array.shape} in {base_stem}")

# Helper functions:
def one_hot_encode(targets, num_classes):
    """
    Convert batch of class indices to one-hot encoded targets.
    Args:
        targets: tensor of shape [batch_size, H, W] containing class indices
        num_classes: int, number of classes in the segmentation task
    Returns:
        One-hot encoded tensor of shape [batch_size, num_classes, H, W]
    """
    batch_size, height, width = targets.size()
    one_hot = torch.zeros(batch_size, num_classes, height, width, device=targets.device)
    return one_hot.scatter_(1, targets.unsqueeze(1), 1)

def get_mask_filenames(row, mask_dir, mask_names, suffix='_cropped'):
    """
    A helper function to get filenames for masks from rows in a dataframe,
    with debug information to ensure correct path construction.
    """
    original_file_name = row['file_name']
    # Removing the specific suffix and replacing with nothing to match the base file name construction.
    file_base = original_file_name.replace('.jpg', '')

    filenames = {}
    # Handle mask filenames
    for mask_name in mask_names:
        mask_file = f"{file_base}_{mask_name}_L.png"
        mask_path = os.path.join(mask_dir, mask_file)

        # Check if the file exists and print the path being checked
        if os.path.isfile(mask_path):
            filenames[mask_name] = mask_path
        else:
            filenames[mask_name] = None  # Or keep as np.nan
            print(f"Checked path not found: {mask_path}")  # Debug output for paths checked

    return pd.Series(filenames)

def get_filename(dl, idx):
    """Helper function to get filenames according to their index in a dataset."""
    fname = dl.dataset.items[idx]
    return os.path.basename(fname)

def get_transform(desired_size):
    """Return a transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((desired_size, desired_size)),  # Resizes the image to the desired size
        transforms.ToTensor()  # Converts the image to a PyTorch tensor
    ])

def process_mask_from_row(row, indices, transform, check_string=None):
    mask = torch.zeros((1, transform.transforms[0].size[0], transform.transforms[0].size[1]), device=device)
    for idx in indices:
        mask_path = row[idx]
        if mask_path and os.path.exists(mask_path) and (check_string is None or check_string not in mask_path):
            mask += transform(Image.open(mask_path)).to(device)[0]
    return mask

def process_mask_for_type(mask: PILMask, mask_type: str, codes: List[str], desired_size: int = 600) -> torch.Tensor:
    """Extract a specific mask type (e.g., infarct, hemorrhage) from a combined mask image and resize."""
    if isinstance(mask, torch.Tensor) and mask.device.type == 'cuda':
        mask = mask.cpu()

    mask_array = np.array(mask)
    mask_val = list(codes).index(mask_type)
    binary_mask = (mask_array == mask_val).astype(np.uint8)

    binary_mask_img = Image.fromarray(binary_mask)
    # print(f"Before Transform - Shape: {binary_mask.shape}, Unique Values: {np.unique(binary_mask)}") # DEBUG
    
    # Apply resize transform to the binary mask image
    transform = get_transform(desired_size)
    resized_mask = transform(binary_mask_img)
    # print(f"After Transform - Shape: {resized_mask.shape}, Unique Values: {np.unique(resized_mask)}") # DEBUG
    
    # Convert the resized mask back to numpy array
    resized_mask_array = np.array(resized_mask)

    # Apply thresholding to make sure the mask is binary
    # Any value greater than 0 should be set to 1
    binary_resized_mask = (resized_mask_array > 0).astype(np.uint8)

    # Convert the binary mask back to a tensor
    binary_mask_tensor = torch.tensor(binary_resized_mask).float()
    return binary_mask_tensor

def create_filename_to_index_mapping(dl):
    """Creates a mapping from filename to DataLoader index."""
    filename_to_index = {}
    for idx in range(len(dl.dataset.items)):
        filename = get_filename(dl, idx)
        filename_to_index[filename] = idx
    return filename_to_index

def get_indices_for_id(df, heart_id, filename_to_index_mapping):
    """Returns a list of DataLoader indices for images associated with a specific heart ID."""
    filenames = df[df['ID'] == heart_id]['file_name'].tolist()
    indices = [filename_to_index_mapping.get(filename) for filename in filenames if filename in filename_to_index_mapping]
    return indices

# Label function:
class LabelFunc:
    def __init__(self, mask_dir, mask_names):
        self.mask_dir = Path(mask_dir)
        self.mask_names = mask_names

    def __call__(self, fn):
        combined_mask = None
        # print(f"Processing file: {fn}")

        for i, mask_name in enumerate(self.mask_names):
            mask_file = self.mask_dir/f"{fn.stem}_{mask_name}_L.png"
            if mask_file.exists():
                mask = Image.open(mask_file).convert('L')
                mask_array = np.array(mask)
                if combined_mask is None:
                    combined_mask = np.zeros(mask_array.shape, dtype=np.uint8)
                combined_mask[mask_array == 255] = i # + 1  # Assign class index, avoiding 0 as it's usually background
            # else:
            #    print(f"Mask file does not exist: {mask_file}")

        if combined_mask is not None:
            return Image.fromarray(combined_mask, mode='L')
        else:
            #print("No masks found, returning empty mask.")
            return Image.fromarray(np.zeros((512, 512), dtype=np.uint8), mode='L')

# loss function
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # This adjustment is done before calling compute_boundaries
        if targets.dim() == 3:  # [batch_size, height, width]
            targets = targets.unsqueeze(1)  # Convert to [batch_size, 1, height, width]
        elif targets.dim() == 2:  # [height, width]
            targets = targets.unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, height, width]

        boundary_maps = self.compute_boundaries(targets)  # Ensure this is now [B, C, H, W]
        if inputs.shape != boundary_maps.shape:
            raise ValueError(f"Shape mismatch between inputs {inputs.shape} and boundary_maps {boundary_maps.shape}")

        loss = F.l1_loss(inputs, boundary_maps)
        return loss

    def compute_boundaries(self, masks):
        # print("Masks shape:", masks.shape) # Debugging statement
        if masks.dim() < 4:
            raise ValueError(f"Expected masks to have 4 dimensions, got {masks.dim()}")
        num_channels = 5  # Assuming 5 channels as before
        batch_size, height, width = masks.shape[0], masks.shape[2], masks.shape[3]
        edge_maps = torch.zeros(batch_size, num_channels, height, width, dtype=torch.float32, device=masks.device)
        for i, mask in enumerate(masks):
            edge_maps[i] = self.sobel_operator(mask)
        return edge_maps

    def sobel_operator(self, mask):
        # Ensure mask is float32 for convolution compatibility
        mask = mask.float()  # Convert mask to float
        # print("Mask shape before Sobel:", mask.shape)  # Debugging statement
        # print("Mask dtype after conversion:", mask.dtype) # Debugging statement
    
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)

        edges_x = F.conv2d(mask, sobel_x, padding=1)
        edges_y = F.conv2d(mask, sobel_y, padding=1)

        edges = torch.sqrt(edges_x**2 + edges_y**2)
        return edges.squeeze(1)  # Reduce back to 3D for further processing if needed

class CombinedCustomLoss(nn.Module):
    def __init__(self, weights, num_classes=5, dice_weight=1, boundary_weight=1, l1_weight=1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(weights, device='cuda'))
        self.dice_loss = DiceLoss()
        self.l1_loss = nn.L1Loss()
        self.boundary_loss = BoundaryLoss()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.l1_weight = l1_weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        if not (0 <= targets).all() and (targets < self.num_classes).all():
            raise ValueError(f"Target labels out of expected range [0, {self.num_classes-1}]")
        
        inputs, targets = inputs.cuda(), targets.cuda()
        ce_loss = self.cross_entropy(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        inputs_softmax = self.softmax(inputs)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        l1_loss = self.l1_loss(inputs_softmax, targets_one_hot)
        boundary_loss = self.boundary_loss(inputs, targets)
        total_loss = ce_loss + self.dice_weight * dice_loss + self.l1_weight * l1_loss + self.boundary_weight * boundary_loss
        return total_loss

# visualization:
class CustomSegmentationInterpretation:
    def __init__(self, learn, dl, preds, targs, losses):
        self.learn = learn
        self.dl = dl
        self.preds = preds
        self.targs = targs
        self.losses = losses

    @classmethod
    def from_learner(cls, learn, dl=None):
        # Explicitly use the validation DataLoader if none is specified
        if dl is None:
            dl = learn.dls.valid
        preds, targs = learn.get_preds(dl=dl, with_decoded=False)
        # Ensure preds and targs are on the same device as the model
        device = next(learn.model.parameters()).device
        losses = []
        for pred, targ in zip(preds, targs):
            pred = pred.unsqueeze(0).to(device)  # Add batch dimension and move to correct device
            targ = targ.unsqueeze(0).to(device)  # Add batch dimension and move to correct device
            loss = learn.loss_func(pred, targ).item()  # Compute loss
            losses.append(loss)
        losses = torch.tensor(losses).to(device)  # Ensure losses tensor is on the correct device
        return cls(learn, dl, preds, targs, losses)

    def top_losses(self, k=None, largest=True):
        losses, idx = self.losses.topk(k if k is not None else len(self.losses), largest=largest)
        return losses, idx    

def resize_mask_to_image(mask, target_shape):
    """Resize mask to match the target image dimensions."""
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def overlay(image_path, row, mask_colors):
    """Overlays masks onto a base image with specified colors."""
    image = mpimg.imread(image_path)
    
    if image.max() > 1:
        image = image / 255.0

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # Use only the RGB channels, ignore alpha if present.

    overlayed_image = image.copy()

    for mask_name, color in mask_colors.items():
        mask_path = row.get(mask_name)
        if pd.notna(mask_path) and mask_path.endswith('.png'):
            try:
                mask = mpimg.imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]  # Use only the first channel if it's a multi-channel image.
                if mask.ndim == 2:
                    mask = mask[..., None]  # Add a channel dimension if missing.
                if mask.max() > 1:
                    mask = mask / 255.0

                mask = resize_mask_to_image(mask, image.shape[:2])
                mask = mask[..., None]  # Ensure mask has three dimensions

                colored_mask = np.zeros_like(image)
                for i in range(3):  # Apply color to RGB channels
                    colored_mask[:, :, i] = mask[:, :, 0] * color[i]

                overlayed_image += colored_mask
                overlayed_image = np.clip(overlayed_image, 0, 1)
            except FileNotFoundError:
                print(f"File not found for mask {mask_name}: {mask_path}")
            except Exception as e:
                print(f"Error processing mask {mask_name}: {e}")
        else:
            print(f"No mask file for {mask_name}")

    # Create legend handles manually
    legend_elements = [mpatches.Patch(facecolor=color[:3], edgecolor='r', label=name.replace('_', ' '))
                       for name, color in mask_colors.items() if pd.notna(row.get(name, None))]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    axs[0].imshow(image)
    axs[0].set_title(os.path.basename(image_path))
    axs[0].axis('off')

    axs[1].imshow(overlayed_image)
    overlay_title = os.path.basename(image_path) + " Overlay"
    axs[1].set_title(overlay_title)
    axs[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def batch_overlay(image_files, batch_size=500):
    """
    Process images in batches to overlay masks.
    """
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        print(f"Processing batch from index {i} to {i + len(batch_files)}")
        for image_file in batch_files:
            image_filename = os.path.basename(image_file)
            image_path = os.path.join(path, image_filename)
            if image_filename in df['file_name'].values:
                row = df[df['file_name'] == image_filename].iloc[0]
                mask_paths = {mask_name: row[mask_name] for mask_name in mask_names if pd.notna(row[mask_name])}
                overlay(image_path, row, mask_colors)
        # Optionally, you can add a break or a condition to wait for user input to proceed to the next batch
        input("Press Enter to continue to the next batch...")

def color_mask(mask, class_to_color, codes):
    """Helper function to color a mask using a class_to_color dictionary."""
    # Create an empty RGBA image
    colored_mask = np.zeros((*mask.shape, 4))  # Adding 4 for RGBA channels
    
    for idx, code in enumerate(codes):
        # Ensure color tuple includes the alpha channel
        color = class_to_color[code]  # Assuming class_to_color already contains RGBA values
        # Apply color where mask has the class index
        colored_mask[mask == idx] = color  
    
    return colored_mask

def show_batch_with_legend(dls, codes, class_to_color, nrows=2, ncols=2, xb=None, yb=None):
    """shows a batch of items in a dataloader with an added legend."""
    if xb is None or yb is None:
        xb, yb = dls.one_batch()

    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols*2, figsize=(12, 6))
    for i in range(nrows):
        for j in range(0, ncols*2, 2):
            # Display image
            axarr[i, j].imshow(xb[i * ncols + j // 2].cpu().permute(1, 2, 0))
            axarr[i, j].axis('off')
            # Display mask with explicit coloring
            mask = yb[i * ncols + j // 2].cpu().numpy()
            colored_mask = color_mask(mask, class_to_color, codes)
            axarr[i, j+1].imshow(colored_mask)
            axarr[i, j+1].axis('off')

    # Add legend
    patches = [mpatches.Patch(color=class_to_color[code], label=code) for code in codes]
    fig.legend(handles=patches, loc='center', ncol=len(codes), bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.show()

def show_img_or_tensor(img_or_tensor, class_to_color=None, codes=None, ax=None, title=None):
    """Visualization function to plot image or tensors as images or their respective masks."""
    if ax is None:
        ax = plt.gca()

    if class_to_color is not None and codes is not None:  # If coloring is requested
        # Convert the mask to its colored version
        if isinstance(img_or_tensor, torch.Tensor):
            mask = img_or_tensor.numpy()
        else:
            mask = img_or_tensor  # It's already a numpy array
        
        img_or_tensor = color_mask(mask, class_to_color, codes)

    if isinstance(img_or_tensor, Image.Image):
        ax.imshow(img_or_tensor)  # Show PIL Image
    elif isinstance(img_or_tensor, torch.Tensor):
        ax.imshow(img_or_tensor.numpy())  # Show Torch tensor
    else:
        ax.imshow(img_or_tensor)  # It's a numpy array

    if title:
        ax.set_title(title, loc='left', wrap=True)
    ax.axis('off')

def create_legend(class_to_color):
    """Create a custom legend using a class_to_color dictionary."""
    legend_patches = [Patch(color=color, label=label) for label, color in class_to_color.items()]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def show_preds(dls, learner, mask_colors, codes, n=6):
    fig, axs = plt.subplots(nrows=n, ncols=2, figsize=(12, n * 6))
    custom_cmap = ListedColormap([mask_colors[code] for code in codes])

    # Fetch a batch
    xb, _ = dls.valid.one_batch()  # Shuffled data

    for i in range(n):
        pred = learner.model(xb[i:i+1]).argmax(dim=1).cpu()  # Prediction for each item in the batch
        input_image = dls.valid.decode((TensorImage(xb[i:i+1]),))[0][0]
        pred_mask = pred[0]

        axs[i, 0].imshow(input_image.permute(1, 2, 0))
        axs[i, 0].set_title('Input Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(pred_mask, cmap=custom_cmap, interpolation='none')
        axs[i, 1].set_title('Predicted Mask')
        axs[i, 1].axis('off')

    plt.tight_layout()
    handles = [Patch(color=mask_colors[label], label=label) for label in codes]
    fig.legend(handles=handles, bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.show()

# calculation of key features:
def calc_features_from_masks(infarct_mask, hemorrhage_mask, area_at_risk_mask):
    """Calculates infarct size segmentation target features"""
    infarct_area = torch.sum(infarct_mask > 0).item()
    hemorrhage_area = torch.sum(hemorrhage_mask > 0).item()
    area_at_risk = torch.sum(area_at_risk_mask > 0).item()
    
    total_infarct = infarct_area + hemorrhage_area
    total_AAR = total_infarct + area_at_risk
    
    if area_at_risk == 0:
        total_infarct = 0
        total_AAR = 0

    infarct_size_per_AAR = (total_infarct / total_AAR) * 100 if total_AAR > 0 else 0
    return infarct_area, hemorrhage_area, area_at_risk, total_infarct, total_AAR, infarct_size_per_AAR

def calculate_roundness(mask):
    """Calculates the roundness of single shapes from a mask area."""
    # Convert the mask from a tensor to a numpy array and ensure it's 8-bit
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Ensure the mask is single-channel
    if mask_np.ndim > 2 and mask_np.shape[0] == 1:
        mask_np = mask_np.squeeze(0)  # Remove channel dimension if it's 1

    # Convert the mask to a binary image (0s and 255s)
    _, binary_mask = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roundness_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        # print(f"Contour - Area: {area}, Perimeter: {perimeter}") # DEBUG
        roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        roundness_list.append(roundness)

    return roundness_list

def calculate_overall_roundness(mask):
    """Calculates the roundness of the entire mask area as a single shape."""
    # Convert the mask from a tensor to a numpy array and ensure it's 8-bit
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Ensure the mask is single-channel
    if mask_np.ndim > 2 and mask_np.shape[0] == 1:
        mask_np = mask_np.squeeze(0)  # Remove channel dimension if it's 1

    # Convert the mask to a binary image (0s and 255s)
    _, binary_mask = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0  # No contours found

    # Combine all contours to treat as a single shape
    all_contours = np.vstack(contours)
    
    area = cv2.contourArea(all_contours)
    perimeter = cv2.arcLength(all_contours, True)
    roundness = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    return roundness

def count_partitions(mask):
    # Convert the mask from a tensor to a numpy array and ensure it's 8-bit
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Ensure the mask is single-channel
    if mask_np.ndim > 2 and mask_np.shape[0] == 1:
        mask_np = mask_np.squeeze(0)  # Remove channel dimension if it's 1

    # Convert the mask to a binary image (0s and 255s)
    _, binary_mask = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"Number of partitions found: {len(contours)}") # DEBUG

    # The number of contours corresponds to the number of partitions
    return len(contours)

def calculate_lv_area(infarct_mask, hemorrhage_mask, area_at_risk_mask, remote_areal_mask):
    lv_area = torch.sum(infarct_mask > 0).item() + torch.sum(hemorrhage_mask > 0).item() + \
              torch.sum(area_at_risk_mask > 0).item() + torch.sum(remote_areal_mask > 0).item()
    return lv_area

# dataloading
def chopper_splitter(items):
    half_index = len(items) // 2
    return IndexSplitter(range(half_index))(items), IndexSplitter(range(half_index, len(items)))(items)

def get_custom_splitter(df):
    # This inner function will be returned and called by IndexSplitter
    def _inner():
        train_idxs = df.index[df['split'] == 'train'].tolist()
        valid_idxs = df.index[df['split'] == 'test'].tolist()
        print(f"Train indices: {train_idxs[:5]}")  # print first few indices
        print(f"Valid indices: {valid_idxs[:5]}")
        return train_idxs, valid_idxs
    return _inner

def get_image_files_func(path):
    """Return only JPEG files from directory."""
    return [p for p in get_image_files(path) if p.suffix in ['.jpg', '.jpeg']]

# predict with the model
def predict_with_model(model_path, learn, dl):
    learn.load(model_path, with_opt=False)  # Load the model without optimizer state
    preds, targs = [], []
    for batch in dl:
        inputs, targets = batch
        # if (targets < 0).any() or (targets >= num_classes).any():
        #    print(f"Invalid targets detected: {targets.unique()}")
        # else:
        with torch.no_grad():
            pred = learn.model(inputs)
        preds.append(pred)
        targs.append(targets)
    return torch.cat(preds, dim=0), torch.cat(targs, dim=0)

# metrics calculations
def aggregate_predictions(dataloader, model):
    model.eval()
    all_probs = []
    all_targs = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_targs.append(yb.cpu())

    return torch.cat(all_probs), torch.cat(all_targs)

def calculate_average_precision(preds, targs, num_classes):
    # Convert predictions to probabilities
    preds_prob = preds.softmax(dim=1)
    APs = []

    for i in range(num_classes):
        # Convert the predictions for class i to a flat numpy array
        preds_i_prob = preds_prob[:, i].reshape(-1).cpu().numpy()

        # Create a binary target array for class i
        targs_i = (targs == i).long().view(-1).cpu().numpy()

        try:
            AP = average_precision_score(targs_i, preds_i_prob)
            APs.append(AP)
        except Exception as e:
            print(f"Error calculating AP for class {i}: {e}")
            APs.append(None)
    
    return APs

def overlap_multi(preds, targs, num_classes, eps=1e-8):
    """Computes the Szymkiewicz-Simpson coefficient for multi-class tensors. """
    total_intersection = 0.0
    total_min_value = 0.0

    for class_idx in range(num_classes):
        # Create binary masks for the specific class
        preds_i = (preds == class_idx).float()
        targs_i = (targs == class_idx).float()

        # Compute the intersection and minimum value between sum of predictions and targets for this class
        intersection = (preds_i * targs_i).sum()
        min_value = min(preds_i.sum(), targs_i.sum())

        total_intersection += intersection
        total_min_value += min_value

    # Compute overall Szymkiewicz-Simpson coefficient
    overall_overlap = (total_intersection + eps) / (total_min_value + eps)
    return overall_overlap

def dice_score_multi(preds, targs, num_classes, eps=1e-8):
    """Computes the Dice score for multi-class tensors.
    
    Args:
        preds (Tensor): A batch of predictions.
        targs (Tensor): A batch of targets.
        num_classes (int): The number of classes.
        eps (float): Epsilon value to prevent division by zero.
        
    Returns:
        float: The average Dice score across all classes.
    """
    dice_scores = 0
    for i in range(num_classes):
        # Create binary masks for this class
        preds_i = (preds == i).float()
        targs_i = (targs == i).float()
        
        # Flatten the tensors
        preds_i = preds_i.view(-1)
        targs_i = targs_i.view(-1)
        
        # Compute the intersection and union
        intersection = (preds_i * targs_i).sum()
        union = preds_i.sum() + targs_i.sum()
        
        # Compute Dice score for this class and accumulate
        dice_scores += (2. * intersection + eps) / (union + eps)
    
    # Average across all classes
    return dice_scores / num_classes

def jaccard_score_multi(preds, targs, num_classes, eps=1e-8):
    """Computes the Jaccard score for multi-class tensors.
    
    Args:
        preds (Tensor): A batch of predictions.
        targs (Tensor): A batch of targets.
        num_classes (int): The number of classes.
        eps (float): Epsilon value to prevent division by zero.
        
    Returns:
        float: The average Jaccard score across all classes.
    """
    jaccard_scores = 0
    for i in range(num_classes):
        # Create binary masks for this class
        preds_i = (preds == i).float()
        targs_i = (targs == i).float()
        
        # Flatten the tensors
        preds_i = preds_i.view(-1)
        targs_i = targs_i.view(-1)
        
        # Compute the intersection and union
        intersection = (preds_i * targs_i).sum()
        union = preds_i.sum() + targs_i.sum() - intersection
        
        # Compute Jaccard score for this class and accumulate
        jaccard_scores += (intersection + eps) / (union + eps)
    
    # Average across all classes
    return jaccard_scores / num_classes

def simple_overlap(preds, targs, class_idx, eps=1e-8):
    """Computes the Szymkiewicz-Simpson coefficient for tensors from a single class."""
    # Ensure predictions are class probabilities and select the class
    preds_i = (preds.argmax(dim=1) == class_idx).float()
    targs_i = (targs == class_idx).float()

    # Flatten the tensors to simplify the operations
    preds_i = preds_i.view(-1)
    targs_i = targs_i.view(-1)

    # Compute the intersection and the minimum value between sum of predictions and targets
    intersection = (preds_i * targs_i).sum()
    min_value = min(preds_i.sum().item(), targs_i.sum().item())

    # Compute Szymkiewicz-Simpson coefficient for this class
    overlap = (intersection + eps) / (min_value + eps)
    return overlap

def dice_score_single(preds, targs, class_idx, eps=1e-8):
    """Computes the Dice score for tensors from a single class."""
    # Create binary masks for the specific class
    preds_i = (preds.argmax(dim=1) == class_idx).float()
    targs_i = (targs == class_idx).float()

    # Flatten the tensors
    preds_i = preds_i.view(-1)
    targs_i = targs_i.view(-1)

    # Compute the intersection and union
    intersection = (preds_i * targs_i).sum()
    union = preds_i.sum() + targs_i.sum()

    # Compute Dice score for this class
    dice_score = (2. * intersection + eps) / (union + eps)
    return dice_score

def jaccard_score_single(preds, targs, class_idx, eps=1e-8):
    """Computes the Jaccard score for tensors from a single class.
    
    Args:
        preds (Tensor): A batch of predictions.
        targs (Tensor): A batch of targets.
        class_idx (int): The index of the respective classes.
        eps (float): Epsilon value to prevent division by zero.
        
    Returns:
        float: The average Jaccard score across all tensors from a class.
    """
    # Create binary masks for the specific class
    preds_i = (preds.argmax(dim=1) == class_idx).float()
    targs_i = (targs == class_idx).float()

    # Flatten the tensors to simplify the operations
    preds_i = preds_i.view(-1)
    targs_i = targs_i.view(-1)

    # Compute the intersection and union
    intersection = (preds_i * targs_i).sum()
    union = preds_i.sum() + targs_i.sum() - intersection

    # Compute Jaccard score for this class
    jaccard_score = (intersection + eps) / (union + eps)
    return jaccard_score

def class_pixel_accuracy(preds, targs, class_idx):
    """
    Calculate pixel accuracy for a single class.

    Args:
    preds (np.ndarray): Predicted labels.
    targs (np.ndarray): True labels.
    class_idx (int): Index of the class to calculate accuracy for.

    Returns:
    float: Pixel accuracy for the specified class.
    """
    # Create binary masks for the specific class
    class_preds = (preds == class_idx).astype(np.uint8)
    class_targs = (targs == class_idx).astype(np.uint8)
    
    # Calculate accuracy
    correct_pixels = np.sum(class_preds == class_targs)
    total_pixels = class_targs.size
    
    return correct_pixels / total_pixels

def weighted_pixel_accuracy(preds, targs, num_classes):
    # Flatten the targets to use in bincount
    targs_flat = targs.view(-1).long()
    # Calculate weights for each class
    weights = torch.bincount(targs_flat, minlength=num_classes)
    weights = weights.float() / targs_flat.numel()
    # Calculate pixel-wise accuracy for each class
    correct = preds.view(-1) == targs_flat
    # Apply weights to the correct predictions
    weighted_correct = weights[preds.view(-1)] * correct.float()
    return weighted_correct.sum().item() / correct.numel()

def average_precision_single(preds, targs, num_classes):
    # Ensure predictions are float before applying softmax
    preds_float = preds.float()  # Convert predictions to float
    preds_prob = preds_float.softmax(dim=1)
    APs = []
    
    for i in range(num_classes):
        # Flatten the tensors for precision calculation
        preds_i = preds_prob[:, i].reshape(-1)
        targs_i = (targs == i).long().reshape(-1)
        # Calculate average precision for this class
        AP = average_precision_score(targs_i.cpu().numpy(), preds_i.cpu().detach().numpy())
        APs.append(AP)

    return APs

def average_precision(preds, targs, num_classes):
    APs = []
    for i in range(num_classes):
        # Flatten the tensors
        preds_i = preds[:, i].view(-1)
        targs_i = (targs == i).long().view(-1)
        AP = average_precision_score(targs_i, preds_i)
        APs.append(AP)
    return APs

def mean_average_precision(APs):
    return sum(APs) / len(APs)

def extract_boundaries(mask):
    eroded_mask = binary_erosion(mask, structure=np.ones((3,3))).astype(np.uint8)
    boundary = mask - eroded_mask
    return boundary

def boundary_f1_score(preds_np, targs_np, num_classes):
    boundary_f1_scores = []
    for i in range(num_classes):
        # Initialize lists to store boundaries for all items in preds and targs
        boundary_preds_list = []
        boundary_targs_list = []
        
        # Loop through all batch items and classes
        for batch_item in range(preds_np.shape[0]):
            preds_i = (preds_np[batch_item] == i).astype(np.uint8)
            targs_i = (targs_np[batch_item] == i).astype(np.uint8)

            boundary_preds_i = extract_boundaries(preds_i)
            boundary_targs_i = extract_boundaries(targs_i)
            
            # Append flattened boundaries for F1 score calculation
            boundary_preds_list.append(boundary_preds_i.flatten())
            boundary_targs_list.append(boundary_targs_i.flatten())

        # Concatenate all batch items for this class and calculate F1 score
        boundary_preds_i_flat = np.concatenate(boundary_preds_list)
        boundary_targs_i_flat = np.concatenate(boundary_targs_list)
        f1 = f1_score(boundary_targs_i_flat, boundary_preds_i_flat)
        
        boundary_f1_scores.append(f1)
    return boundary_f1_scores

def calculate_metrics(preds, targs, num_classes):
    preds_np = preds.argmax(dim=1).cpu().numpy()
    targs_np = targs.cpu().numpy()
    scores = []

    for i in range(num_classes):
        # Metrics
        dice = dice_score_single(preds, targs, i)
        jaccard = jaccard_score_single(preds, targs, i)
        pixel_acc = class_pixel_accuracy(preds_np, targs_np, i)
        overlap = simple_overlap(preds, targs, i)
        boundary_f1 = boundary_f1_score(preds_np, targs_np, num_classes)[i]

        scores.append({
            "Class": classes[i],
            "Dice": dice.item(),
            "Jaccard": jaccard.item(),
            "Pixel Accuracy": pixel_acc,
            "Overlap": overlap.item(),
            "Boundary F1": boundary_f1,
        })

    return pd.DataFrame(scores)

def calculate_multiclass_metrics(preds, targs, num_classes):
    """ Calculate various metrics across multiple classes. """
    preds_classes = preds.argmax(dim=1)
    overlap = overlap_multi(preds_classes, targs, num_classes)
    dice = dice_score_multi(preds_classes, targs, num_classes)
    jaccard = jaccard_score_multi(preds_classes, targs, num_classes)
    weighted_acc = weighted_pixel_accuracy(preds_classes, targs, num_classes)
    return overlap, dice, jaccard, weighted_acc

def calculate_mean_average_precision(preds, targs, num_classes):
    """ Calculate mean average precision across all classes. """
    APs = []
    for i in range(num_classes):
        preds_i = preds[:, i].reshape(-1)
        targs_i = (targs == i).long().reshape(-1)
        AP = average_precision_score(targs_i.cpu().numpy(), preds_i.cpu().numpy())
        APs.append(AP)
    mAP = np.mean(APs)
    return mAP

def calculate_boundary_f1(preds_np, targs_np, num_classes):
    """ Calculate boundary F1 scores and return their average. """
    boundary_f1_scores = boundary_f1_score(preds_np, targs_np, num_classes)
    overall_boundary_f1 = np.mean(boundary_f1_scores)
    return overall_boundary_f1

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# GPU support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% step 0.5 Preprocessing
# crop images and labels to their respective 
mask_names = ['infarct', 'lumen', 'epicardium', 'right_ventricle', 'remote_areal', 
              'hemorrhage', 'area_at_risk_without_infarct,_hemorrhage', 'remaining_areas']

im_dir = ''
mask_dir = ''
out_dir = ''

crop_files(im_dir, mask_dir, mask_names, out_dir)

# %% Opt: remove file extensions
# OPTIONAL: remove file extensions
# Replace 'target_directory_path' with the path to your target directory
rename_dir_path = ''

# Loop through all the files in the target directory
for filename in os.listdir(rename_dir_path):
    # Check if the filename ends with '_cropped.out'
    if filename.endswith('_cropped.out.jpg'):
        # Construct the full path to the current file
        old_file_path = os.path.join(rename_dir_path, filename)
        
        # Remove '_cropped.out' from the filename
        new_filename = filename.replace('_cropped.out', '')
        
        # Construct the full path to the new file
        new_file_path = os.path.join(rename_dir_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        
        # Print a message indicating that the file has been renamed
        print(f'Renamed "{old_file_path}" to "{new_file_path}"')

# %% Opt: copy files
# OPTIONAL: move masks to destination directory
im_dir = '/'
mask_dir = ''

# Loop through all the files in the images directory
for image_filename in os.listdir(im_dir):
    # Check if the file is a JPG image
    if image_filename.endswith('.jpg'):
        # Extract the stem (filename without extension) from the image filename
        filename_stem = os.path.splitext(image_filename)[0]
        
        # Look for corresponding mask and bounding box files in the masks directory
        for mask_filename in os.listdir(mask_dir):
            # Check if the mask filename contains the image's stem and ends with .png (labels) or .txt (bounding boxes)
            if mask_filename.startswith(filename_stem) and (mask_filename.endswith('_L.png') or mask_filename.endswith('_bbox.txt')):
                # Construct the full path to the mask or bounding box file in the masks directory
                mask_file_path = os.path.join(mask_dir, mask_filename)
                
                # Construct the destination path in the images directory
                dest_path = os.path.join(im_dir, mask_filename)
                
                # Copy the mask or bounding box file to the images directory
                shutil.copy2(mask_file_path, dest_path)
                
                # Print a message indicating that the file has been copied
                print(f'Copied "{mask_file_path}" to "{dest_path}"')

# %% Merging masks for 5 parameter model
src_folder = ''
dest_folder = ''

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

for root, _, files in os.walk(src_folder):
    grouped_files = {}
    for file in files:
        lower_file = file.lower()
        if file.endswith('.jpg'):
            base_name = file[:-4]  # Remove the .jpg extension for base images
        else:
            base_name = '_'.join(file.split('_')[:4])  # Use the first four parts for masks
        grouped_files.setdefault(base_name, []).append(file)

    for base_name, group_files in grouped_files.items():
        aar_masks = []
        remaining_masks = []
        base_image_size = None
        output_paths = {}

        for file_name in group_files:
            full_path = os.path.join(root, file_name)
            if file_name.lower().endswith('.jpg'):
                shutil.copy(full_path, os.path.join(dest_folder, file_name))
                base_image_size = Image.open(full_path).size

            elif 'hemorrhage_L.png' in file_name or 'area_at_risk_without_infarct,_hemorrhage_L.png' in file_name:
                aar_masks.append(full_path)
                output_paths['AAR'] = os.path.join(dest_folder, f"{base_name}_AAR_L.png")

            elif 'remote_areal_L.png' in file_name:
                new_path = os.path.join(dest_folder, f"{base_name}_remote_L.png")
                shutil.copy(full_path, new_path)

            elif 'right_ventricle_L.png' in file_name:
                new_path = os.path.join(dest_folder, f"{base_name}_right_ventricle_L.png")
                shutil.copy(full_path, new_path)

            elif 'infarct_L.png' in file_name:
                new_path = os.path.join(dest_folder, f"{base_name}_infarct_L.png")
                shutil.copy(full_path, new_path)

            elif 'lumen_L.png' in file_name or 'epicardium_L.png' in file_name or 'remaining_areas_L.png' in file_name:
                remaining_masks.append(full_path)
                output_paths['remaining'] = os.path.join(dest_folder, f"{base_name}_remaining_L.png")

        if aar_masks and base_image_size:
            merge_masks(aar_masks, output_paths['AAR'], base_image_size)
        if remaining_masks and base_image_size:
            merge_masks(remaining_masks, output_paths['remaining'], base_image_size)

print("Processing complete.")

# %% # check for shape mismatch
mask_names = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']
path = ''
mismatches = check_mask_dimensions(path, path, mask_names)
for mismatch in mismatches:
    print(f"Base image and mask size mismatch: {mismatch}")

# %% mismatch handling
# for dimension mismatch use this pre-processing step:
base_image_path = ''
mask_dir = ''
mask_names = ['infarct', 'AAR', 'remote', 'right_ventricle', 'remaining']
correct_mask_dimensions(base_image_path, mask_dir, mask_names)

# %% Step 1: set up dataframe
# set up the path and dataframe
path = ''
file_name = 'rat_regional_IR_testset.xlsx'
# Load base data without split information
df = pd.read_excel(path + '/' + file_name, header=None, usecols=range(5))
df.columns = ['file_name', 'protocol', 'ID', 'slice_number', 'slice_side']

split_column = pd.read_excel(path + '/' + file_name, header=None, usecols=[5])

# Add the split column to the DataFrame
df['split'] = split_column

# construct dataframe with masks
# add path for masks to the dataframe
# Caution: Just run once!
# Your original DataFrame 'df' and other setup code
mask_names = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']
mask_dir = path

# Define the suffix used in the file renaming
#suffix = '_cropped'

# Update the file_name column to include the _cropped suffix for images
#df['file_name'] = df['file_name'].str.replace('.jpg', f'{suffix}.out.jpg')

# Apply the updated function to generate mask filenames
mask_filenames_df = df.apply(lambda row: get_mask_filenames(row, mask_dir, mask_names), axis=1)

# Concatenate the mask filenames with the original DataFrame
df = pd.concat([df, mask_filenames_df], axis=1)

print(df.head(20))

# %% Step 1.5 Visulization of base images and masks
# attach masks to original images for visualization:
# Create a dictionary of colors, skipping the first color in the palette
mask_colors = {
    'infarct': (1, 1, 1),  # white
    'AAR': (0.6, 0.4, 0),  # light red
    'remote': (1, 0, 0),  # red
    'right_ventricle': (0, 0, 1),  # blue
    'remaining': (0, 0.647, 0)  # green
    }

# Example usage
image_filename = 'IPC_22082024R1green_5_a.jpg'
image_path = os.path.join(path, image_filename)
row = df[df['file_name'] == image_filename].iloc[0]

overlay(image_path, row, mask_colors)

# %% check all images with overlay
# Assuming 'path' is your directory containing the images
image_files = glob.glob(os.path.join(path, '*.jpg'))  # or '*.png' if your images are in png format

batch_overlay(image_files, batch_size=200)

# %% Step 2: calculate target key features: 
## calculate total infarct, total area at risk and infarct size per AAR
# Define columns for different mask types
infarct_col_indices = [idx for idx, col in enumerate(df.columns) if 'infarct' in col]
hemorrhage_col_indices = [idx for idx, col in enumerate(df.columns) if 'hemorrhage' in col]
area_at_risk_col_indices = [idx for idx, col in enumerate(df.columns) if 'area_at_risk_without_infarct,_hemorrhage' in col]
remote_areal_col_indices = [idx for idx, col in enumerate(df.columns) if 'remote_areal' in col]
rv_col_indices = [idx for idx, col in enumerate(df.columns) if 'right_ventricle' in col]

results = []

# apply size transformation from training segment
DESIRED_SIZE = 512
transform = get_transform(DESIRED_SIZE)

# iterate through the dataframe with ground truth masks to calculate target features
for index, row in df.iterrows():
    filename = row['file_name']
    protocol = row['label']
    if protocol not in ['placebo', 'Protect']:
        continue

    # process masks 
    infarct_mask = process_mask_from_row(row, infarct_col_indices, transform, "area_at_risk_without_infarct,_hemorrhage")
    hemorrhage_mask = process_mask_from_row(row, hemorrhage_col_indices, transform, "area_at_risk_without_infarct,_hemorrhage")
    area_at_risk_mask = process_mask_from_row(row, area_at_risk_col_indices, transform)
    remote_areal_mask = process_mask_from_row(row, remote_areal_col_indices, transform)
    rv_mask = process_mask_from_row(row, rv_col_indices, transform)

    # calculate key features
    features = calc_features_from_masks(infarct_mask, hemorrhage_mask, area_at_risk_mask)
    
    # Calculate partitions and LV area
    partitions = count_partitions(infarct_mask)
    lv_area = calculate_lv_area(infarct_mask, hemorrhage_mask, area_at_risk_mask, remote_areal_mask)
    
    # Calculate RV area
    rv_area = torch.sum(rv_mask > 0).item()

    # calculate roundness values
    roundness_single = calculate_roundness(infarct_mask)  # List of roundness for each partition
    roundness_av = np.mean([float(rnd) for rnd in roundness_single]) if roundness_single else 0
    roundness_overall = calculate_overall_roundness(infarct_mask)  # Roundness for combined shape

    # Append all calculated features to results
    results.append([filename, protocol] + list(features) + 
                   [lv_area, rv_area, partitions, roundness_single, roundness_av, roundness_overall])

# Define columns for DataFrame
columns = ['filename', 'protocol', 'infarct_area', 'hemorrhage_area', 'area_at_risk', 
           'total_infarct', 'total_AAR', 'infarct_size_per_AAR', 'lv_area', 'rv_area',
           'partitions', 'roundness_single', 'roundness_av', 'roundness_overall' ]
results_df = pd.DataFrame(results, columns=columns)

# %% dataframe and saving
# Displaying the results
print(results_df)

# Save the results DataFrame to an Excel file
results_df.to_excel('traget features segmentation.xlsx', index=True)

# %% Visualize the target features
# Setting the style
sns.set_style("whitegrid")

# List of metrics to be plotted
metrics = ['infarct_area', 'hemorrhage_area', 'area_at_risk', 'total_infarct', 'total_AAR', 'infarct_size_per_AAR']

# Plotting each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='protocol', y=metric, palette="colorblind")
    plt.title(f'Boxplot of {metric} by Protocol')
    plt.show()

# %%
# Additional metrics to be plotted
additional_metrics = ['roundness_av', 'roundness_overall', 'partitions', 'lv_area']

# Plotting each additional metric
for metric in additional_metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='protocol', y=metric, palette="colorblind")
    plt.title(f'Boxplot of {metric} by Protocol')
    plt.show()

# %% Step 3: test with a pretrained image segmentation model:
## Dataloading:
# path to the data, the ground truth original images, class labels and bounding boxes
data = Path(path)

images = get_image_files_func(data)

# define class names ("codes") for the segmentation classes
mask_names = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']

# create an instance of LabelFunc class
label_func = LabelFunc(data, mask_names)

# get train/valid split from dataframe
split_idx = (df['split'] == 'train').astype(int)

SIZE = 384

transforms = [RandomResizedCrop(SIZE, min_scale=1, ratio=(1, 1))]

# Ensure the `DataBlock` correctly applies transformations and retrieves labels
dblock = DataBlock(
    blocks=(ImageBlock, MaskBlock(mask_names)),
    get_items=get_image_files_func,
    get_y=label_func,
    splitter=IndexSplitter(np.where(split_idx == 0)[0]), 
    item_tfms=Resize(SIZE, method=ResizeMethod.Pad, pad_mode='zeros'),
    batch_tfms=transforms
)

# Create DataLoaders and test them
dls = dblock.dataloaders(data, bs=8, val_shuffle=False)

# %% verify loaded data (1)
dblock.summary(data)

print(len(dls.train_ds))
print(len(dls.valid_ds))

# %% verify loaded data (2)
# Example to check the first few mask paths
for i, row in df.head(20).iterrows():
    print(f"Checking paths for file: {row['file_name']}")
    for mask_name in mask_names:
        mask_path = row.get(mask_name, None)
        if mask_path:
            exists = Path(mask_path).exists()
            print(f"Path for {mask_name}: {mask_path}, Exists: {exists}")
        else:
            print(f"No path for {mask_name}")

# %% check loaded data (1)
# check loaded data
dls.valid.show_batch(unique=False, max_n=8) # unique=True for augmentations on one image

# %% check loaded data (2)
# show batch with legend to be sure that all the labeles are correctly applied
# Define a custom colormap
tab20_colors = plt.cm.tab20.colors[:9]  # first 9 colors from matplotlib
class_to_color = {name: tab20_colors[i] + (1.0,) for i, name in enumerate(mask_names)} # add to color dictionary

# use the costum show_batch function with the color dictionary
show_batch_with_legend(dls, mask_names, class_to_color, nrows=2, ncols=2)

# %% check loaded data (3)
# to comapre show_batch and the costum one with the same batch
batch = dls.one_batch()
xb, yb = batch

# Show using standard show_batch
dls.show_batch(b=(xb, yb), nrows=2, ncols=2)

# Show using custom show_batch_with_legend
show_batch_with_legend(dls, mask_names, class_to_color, nrows=2, ncols=2, xb=xb, yb=yb)


# %% visualize first mask in dataloader
# Sample visualization of masks from DataLoader
xb, yb = next(iter(dls.train))
sample_mask = yb[0].cpu()  # Taking the first mask from the batch

plt.imshow(sample_mask, cmap='gray')
plt.title("Sample Mask from DataLoader")
plt.colorbar()
plt.show()

# %% define model parameters
# load last calculated normalized class weights
normalized_weights = [0.1, 0.6, 0.4, 0.1, 0.2]

# build the learner
dice = DiceMulti() # define metrics, here Dice coefficient

# specifiy class weights for the Loss-function
weights_tensor = torch.tensor(normalized_weights).cuda()
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
custom_loss = CombinedCustomLoss(weights=normalized_weights)

# create the learner for dynamic U-Net
learn = unet_learner(dls, resnet34, 
                     metrics=[foreground_acc, dice],
                     opt_func=Adam, 
                     loss_func=custom_loss)

# %%
# empty cache
torch.cuda.empty_cache()

# %% predict
# predict with the models
learn.model.cuda()  # move to GPU

# List of model paths of cross-validation
model_path = '' + '/'
fold_one = model_path + ''
fold_two = model_path + ''
fold_three = model_path + ''
fold_four = model_path + ''             
fold_five = model_path + ''

num_classes = 5

# Predict using one specified model fold
# Perform predictions for the first part of the dataset
preds, targs = predict_with_model(fold_one, learn, dls.valid)

# Clear memory after predictions if needed
torch.cuda.empty_cache()

#%% DEBUG: shape (2)
# DEBUG
test_batch = next(iter(dls.train))
xb, yb = test_batch
preds = learn.model(xb)  # assuming learn is your Learner object
print("Output shape from model:", preds.shape)
print("First few model outputs:", preds[0])

# %% show results custom function
# show results
# Define your mask colors
mask_colors = {
    'remaining': (0, 0, 0),          # Black
    'infarct': (1, 1, 1),            # White
    'AAR': (1, 0.5, 0.5),            # Light Red
    'remote': (1, 0, 0),             # Red
    'right_ventricle': (0, 0, 1)     # Blue
}

# Class codes sequence
codes = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']

# Usage
show_preds(dls, learn, mask_colors, codes, n=5)
torch.cuda.empty_cache()

# %% Step 4: calculate metrics:
# calculate metrics per class in the segmentation
# List of class names
# Usage
classes = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']
num_classes = 5

preds_agg, targs_agg = aggregate_predictions(dls.valid, learn.model)
torch.cuda.empty_cache()

scores_df = calculate_metrics(preds_agg, targs_agg, num_classes)
torch.cuda.empty_cache()

APs = calculate_average_precision(preds_agg, targs_agg, num_classes)
torch.cuda.empty_cache()

scores_df['APs'] = APs
print(scores_df)

# %%
# metrics overall
# Aggregate predictions and targets
preds_agg, targs_agg = aggregate_predictions(dls.valid, learn.model)

# Calculate multi-class metrics
overlap, dice, jaccard, weighted_acc = calculate_multiclass_metrics(preds_agg, targs_agg, len(classes))

# Calculate mean average precision
mAP = calculate_mean_average_precision(preds_agg, targs_agg, len(classes))

# Calculate overall boundary F1 score
preds_np = preds_agg.argmax(dim=1).cpu().numpy()
targs_np = targs_agg.cpu().numpy()
overall_boundary_f1 = calculate_boundary_f1(preds_np, targs_np, len(classes))

# %%
# Convert TensorMask or similar to standard numeric types
overlap = overlap.item() if hasattr(overlap, 'item') else float(overlap)
dice = dice.item() if hasattr(dice, 'item') else float(dice)
jaccard = jaccard.item() if hasattr(jaccard, 'item') else float(jaccard)

metrics = {
    "Metric": ["Weighted Pixel Accuracy", "Overall Overlap", "Dice Score", "Jaccard Score", "Mean Average Precision", "Overall Boundary F1 Score"],
    "Value": [weighted_acc, overlap, dice, jaccard, mAP, overall_boundary_f1]
}

# Convert dictionary to DataFrame
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# %% Write as excel
# Specify the Excel file name
file_name = 'segm4_RAT_testset_metrics_cv1.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter(file_name) as writer:
    # Write each dataframe to a different sheet
    scores_df.to_excel(writer, sheet_name='Single Class Metrics', index=False)
    metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)

print(f"Metrics saved to {file_name}")

# %% Step 5: Interpretation and calculation of key features
# get interpretation data as an object
interp = CustomSegmentationInterpretation.from_learner(learn)

torch.cuda.empty_cache()

# %%
# Define functions, color
# Define your class labels and corresponding colors here (make sure they're RGBA)
mask_colors = {
    'remaining': (0, 0, 0, 1),        # Black
    'infarct': (255, 255, 255, 1),    # White
    'AAR': (255, 128, 128, 1),        # Light Red
    'remote': (255, 0, 0, 1),         # Red
    'right_ventricle': (0, 0, 255, 1) # Blue
}

# Normalize colors to range [0, 1] for matplotlib
mask_colors = {k: tuple(c / 255 if i < 3 else c for i, c in enumerate(v)) for k, v in mask_colors.items()}
classes = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']

# %% check individual images from the valid dataset
idx = 10   # get an image from the valid dataset 

# Get the original image, target mask, and prediction
input_image = learn.dls.valid.dataset[idx][0]  
target_mask = targs[idx].cpu()  
pred_mask = preds[idx].argmax(dim=0).cpu()  

# Get the filename
filename = get_filename(learn.dls.valid, idx)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

# Plot original image
show_img_or_tensor(input_image, ax=axs[0], title=f'Input Image:\n{filename}')
# Plot target mask
show_img_or_tensor(target_mask, mask_colors, classes, ax=axs[1], title='Target Mask')
# Plot predicted mask
show_img_or_tensor(pred_mask, mask_colors, classes, ax=axs[2], title='Predicted Mask')

# Create a legend for the masks
create_legend(mask_colors)

plt.tight_layout()
plt.show()

# %%
# visualize all images for specific ID
# Use the function to create the mapping for the validation DataLoader
filename_to_index_mapping = create_filename_to_index_mapping(learn.dls.valid)

# Example usage:
heart_id = "01032023R1purple"  # Specify the heart ID you're interested in
indices = get_indices_for_id(df, heart_id, filename_to_index_mapping)

# Visualize the results for each index
for idx in indices:
    input_image = learn.dls.valid.dataset[idx][0]
    target_mask = targs[idx].cpu()
    pred_mask = preds[idx].argmax(dim=0).cpu()
    filename = get_filename(learn.dls.valid, idx)

    # Plotting setup
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    show_img_or_tensor(input_image, ax=axs[0], title=f'Input Image:\n{filename}')
    show_img_or_tensor(target_mask, mask_colors, classes, ax=axs[1], title='Target Mask')
    show_img_or_tensor(pred_mask, mask_colors, classes, ax=axs[2], title='Predicted Mask')
    plt.tight_layout()
    plt.show()

# %% Top losses
# Get the top k losses and their indices
k = 30
top_losses, top_idxs = interp.top_losses(k=k)

# Iterate over the top losses
for i in range(k):
    idx = top_idxs[i]
    actual = targs[idx]  
    pred = preds[idx]
    loss = top_losses[i]

    filename = get_filename(learn.dls.valid, idx)

    print(f"Top {i+1} Loss: {loss.item()}, {filename}")

# Visualize the top losses
for i in range(k):
    idx = top_idxs[i]

    # Get the original image, target mask, and prediction
    input_image = learn.dls.valid.dataset[idx][0]  # This fetches the input image (PIL Image)
    target_mask = targs[idx].cpu()  # Targets are here, moved to CPU
    pred_mask = preds[idx].argmax(dim=0).cpu()  # Get the most likely class for each pixel, moved to CPU

    # Get the filename
    filename = get_filename(learn.dls.valid, idx)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

    # Plot original image
    show_img_or_tensor(input_image, ax=axs[0], title=f'Input Image:\n{filename}')
    # Plot target mask
    show_img_or_tensor(target_mask, mask_colors, classes, ax=axs[1], title='Target Mask')
    # Plot predicted mask
    show_img_or_tensor(pred_mask, mask_colors, classes, ax=axs[2], title='Predicted Mask')

    # Create a legend for the masks
    create_legend(mask_colors)

    plt.tight_layout()
    plt.show()

# %% Lowest Losses
# Get the lowest k losses and their indices
k = 20
lowest_losses, lowest_idxs = interp.top_losses(k=k, largest=False)

# Iterate over the lowest losses
for i in range(k):
    idx = lowest_idxs[i]
    actual = targs[idx]  
    pred = preds[idx]
    loss = lowest_losses[i]

    filename = get_filename(learn.dls.valid, idx)

    print(f"Top {i+1} Loss: {loss.item()}, {filename}")

## Visualize the lowest losses
for i in range(k):
    idx = lowest_idxs[i]

    # Get the original image, target mask, and prediction
    input_image = learn.dls.valid.dataset[idx][0]  # This fetches the input image (PIL Image)
    target_mask = targs[idx].cpu()  # Targets are here, moved to CPU
    pred_mask = preds[idx].argmax(dim=0).cpu()  # Get the most likely class for each pixel, moved to CPU

    # Get the filename
    filename = get_filename(learn.dls.valid, idx)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns

    # Plot original image
    show_img_or_tensor(input_image, ax=axs[0], title=f'Input Image:\n{filename}')
    # Plot target mask
    show_img_or_tensor(target_mask, mask_colors, classes, ax=axs[1], title='Target Mask')
    # Plot predicted mask
    show_img_or_tensor(pred_mask, mask_colors, classes, ax=axs[2], title='Predicted Mask')

    # Create a legend for the masks
    create_legend(mask_colors)

    plt.tight_layout()
    plt.show()

# %% Calculate key features from the valid dataset (1. ground truth, 2. predictions)
# Calculate features from ground truth in the valid dataset
DESIRED_SIZE = 384  # Set desired size for transforms

codes = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']

# Calculate key features from ground truth in the validation dataset
columns = ['filename', 'infarct_area', 'aar_area', 'rv_area', 'remote_area', 
           'infarct_size_per_aar', 'lv_area', 'roundness_single', 'roundness_avg', 'roundness_overall', 'partitions']
targets_df = pd.DataFrame(columns=columns)

total_items = len(dls.valid.dataset.items)
batch_size = dls.valid.bs

# Loop for calculation of target features
for idx, (x, target_batch) in enumerate(dls.valid):
    # print(f"Processing batch {idx+1}/{len(dls.valid)} with size {len(target_batch)}")
    batch_results = []
    for target_idx, target in enumerate(target_batch):
        actual_index = idx * batch_size + target_idx
        if actual_index >= total_items:
            break  # Prevent processing beyond the number of available items

        filename = dls.valid.dataset.items[actual_index].name
        # print(f"Item index {actual_index}, Filename: {filename}")
        
        # Process each mask type
        infarct_mask = (target == 1).float()  # Binary mask for 'infarct'
        aar_mask = (target == 2).float()  # Binary mask for 'AAR'
        remote_mask = (target == 3).float()  # Binary mask for 'remote'
        rv_mask = (target == 4).float()  # Binary mask for 'right_ventricle'

        # Calculate features
        infarct_area = torch.sum(infarct_mask).item()
        aar_area = torch.sum(aar_mask).item()
        remote_area = torch.sum(remote_mask).item()
        rv_area = torch.sum(rv_mask).item()
        lv_area = infarct_area + aar_area + remote_area  # Total LV area
        infarct_size_per_aar = (infarct_area / (aar_area + infarct_area)) * 100 if (aar_area + infarct_area) > 0 else 0

        # Calculate roundness and partitions
        roundness_single = calculate_roundness(infarct_mask)
        roundness_avg = np.mean(roundness_single) if roundness_single else 0
        roundness_overall = calculate_overall_roundness(infarct_mask)
        partitions = count_partitions(infarct_mask)

        # Prepare data row
        data_row = {
            'filename': filename,
            'infarct_area': infarct_area,
            'aar_area': aar_area,
            'rv_area': rv_area,
            'remote_area': remote_area,
            'infarct_size_per_aar': infarct_size_per_aar,
            'lv_area': lv_area,
            'roundness_single': roundness_single,
            'roundness_avg': roundness_avg,
            'roundness_overall': roundness_overall,
            'partitions': partitions
        }
        batch_results.append(data_row)
    
    # Convert batch results to DataFrame and concatenate with the existing DataFrame
    batch_df = pd.DataFrame(batch_results)
    targets_df = pd.concat([targets_df, batch_df], ignore_index=True)

print(targets_df)

# %% Calculate key features from the predictions
# Calculate key features from predictions
DESIRED_SIZE = 384  # set desired size for transforms

preds_df = pd.DataFrame(columns=columns)

# Get predictions and convert from probability maps to hard masks
preds, _ = learn.get_preds(dl=dls.valid)
pred_masks = preds.argmax(dim=1)  # Convert probabilities to class indices

# loop for calculation of features from predictions
for idx, pred_mask in enumerate(pred_masks):
    # Prepare to collect batch results if needed
    batch_results = []
    
    # Process each predicted mask
    infarct_mask = (pred_mask == 1).float()
    aar_mask = (pred_mask == 2).float()
    remote_mask = (pred_mask == 3).float()
    rv_mask = (pred_mask == 4).float()

    # Calculate features
    infarct_area = torch.sum(infarct_mask).item()
    aar_area = torch.sum(aar_mask).item()
    remote_area = torch.sum(remote_mask).item()
    rv_area = torch.sum(rv_mask).item()
    lv_area = infarct_area + aar_area + remote_area  # Total LV area
    infarct_size_per_aar = (infarct_area / (aar_area + infarct_area)) * 100 if (aar_area + infarct_area) > 0 else 0

    # Calculate roundness and partitions
    roundness_single = calculate_roundness(infarct_mask)
    roundness_avg = np.mean(roundness_single) if roundness_single else 0
    roundness_overall = calculate_overall_roundness(infarct_mask)
    partitions = count_partitions(infarct_mask)

    data_row = {
        'filename': dls.valid.dataset.items[idx].name,
        'infarct_area': infarct_area,
        'aar_area': aar_area,
        'rv_area': rv_area,
        'remote_area': remote_area,
        'infarct_size_per_aar': infarct_size_per_aar,
        'lv_area': lv_area,
        'roundness_single': roundness_single,
        'roundness_avg': roundness_avg,
        'roundness_overall': roundness_overall,
        'partitions': partitions
    }

    # Append to the batch results
    batch_results.append(data_row)

    # Convert batch results to DataFrame and concatenate with the existing DataFrame
    batch_df = pd.DataFrame(batch_results)
    preds_df = pd.concat([preds_df, batch_df], ignore_index=True)

print(preds_df)

# %%
# store the dataframes in an excel file
with pd.ExcelWriter('segm4_RAT_testset_features.xlsx') as writer:
    targets_df.to_excel(writer, sheet_name='targets', index=False)
    preds_df.to_excel(writer, sheet_name='predicitions', index=False)

# Add a new column 'Type' to each dataframe
targets_df['Type'] = 'targets'
preds_df['Type'] = 'predicitions'

# Concatenate the two dataframes
features_df = pd.concat([targets_df, preds_df], ignore_index=True)

# %% Visualization:
# Visualize targets and predictions in boxplots
targets_df['protocol'] = targets_df['filename'].apply(lambda x: x[:7])
preds_df['protocol'] = preds_df['filename'].apply(lambda x: x[:7])

# Combine the dataframes
targets_df['Type'] = 'Target'
preds_df['Type'] = 'Predicted'
features_df = pd.concat([targets_df, preds_df], ignore_index=True)
print(features_df)

# %% Visualization:
# Visualize targets and predictions in boxplots
sns.set_style("whitegrid")  # Set style

# Define metrics
metrics = ['infarct_area', 'aar_area', 'rv_area', 'remote_area', 
           'infarct_size_per_aar', 'lv_area', 'roundness_avg', 
           'roundness_overall', 'partitions']

# Define the order of protocols based on the expected data
order = ['placebo', 'Protect']  # Replace with actual protocol names

# Plot each metric
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=features_df, x='protocol', y=metric, hue='Type')
    plt.title(f'Boxplot of {metric} by Protocol (Target vs Predicted)')
    plt.legend(loc='upper right')
    plt.show()

# %%
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.lines as mlines
import seaborn as sns

def plot_correlation(data, subset_title):
    if data.empty:
        print(f"No data available for {subset_title}.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='infarct_size_per_AAR_Targets', y='infarct_size_per_AAR_Predicted', data=data)
    r, _ = pearsonr(data['infarct_size_per_AAR_Targets'], data['infarct_size_per_AAR_Predicted'])
    mae = mean_absolute_error(data['infarct_size_per_AAR_Targets'], data['infarct_size_per_AAR_Predicted'])
    rmse = np.sqrt(mean_squared_error(data['infarct_size_per_AAR_Targets'], data['infarct_size_per_AAR_Predicted']))
    max_value = max(data['infarct_size_per_AAR_Targets'].max(), data['infarct_size_per_AAR_Predicted'].max())
    plt.plot([0, max_value], [0, max_value], 'k--', label='Line of Identity')
    m, b = np.polyfit(data['infarct_size_per_AAR_Targets'], data['infarct_size_per_AAR_Predicted'], 1)
    plt.plot(data['infarct_size_per_AAR_Targets'], m * data['infarct_size_per_AAR_Targets'] + b, 'r--', label=f'Correlation Line (r: {r:.2f})')
    plt.xlabel('Targets - Infarct size [%AAR]')
    plt.ylabel('Predictions - Infarct size [%AAR]')
    plt.legend(title=f'{subset_title} - MAE: {mae:.2f}, RMSE: {rmse:.2f}', loc='lower right')
    plt.show()

# Extract data from both DataFrames
correl = {
    'infarct_size_per_AAR_Targets': targets_df['infarct_size_per_aar'],
    'infarct_size_per_AAR_Predicted': preds_df['infarct_size_per_aar'],
    'protocol': targets_df['protocol']
}

# Convert to DataFrame
correl_df = pd.DataFrame(correl)
placebo_data = correl_df[correl_df['protocol'].str.lower() == 'placebo']
protect_data = correl_df[correl_df['protocol'].str.lower() == 'protect']

plot_correlation(correl_df, 'All Data, fold 1')
plot_correlation(placebo_data, 'Placebo, fold 1')
plot_correlation(protect_data, 'Protect, fold 1')

# %% find items with high predicted infarct size, which are actual 0%
# Define a threshold for severe misses 
threshold = 5  # Percent as unit

# Merge the dataframes on 'filename'
merged_df = targets_df.merge(preds_df, on='filename', suffixes=('_Targets', '_Predicted'))

# Adjust the filtering condition to match the actual column names
# Assuming 'infarct_size_per_aar' is the correct column name and it's now suffixed properly after merge
misses_df = merged_df[(merged_df['infarct_size_per_aar_Predicted'] > threshold) & (merged_df['infarct_size_per_aar_Targets'] == 0)]

# Extract the filenames and infarct sizes for further analysis
misses_info = misses_df[['filename', 'infarct_size_per_aar_Targets', 'infarct_size_per_aar_Predicted', 'protocol_Targets', 'protocol_Predicted']]

# Display the extracted information
print(misses_info)

# %% Visualize the highest misses
# Visualize the highest misses for target feature infarct size
highest_misses = misses_info['filename'].tolist()
valid_filenames = [get_filename(learn.dls.valid, i) for i in range(len(learn.dls.valid.dataset))]
high_miss_fns = misses_info['filename'].tolist()
high_miss_idxs = [valid_filenames.index(fname) for fname in high_miss_fns if fname in valid_filenames]

## Visualize the highest misses 
k = len(highest_misses)

for i in range(k):
    idx = high_miss_idxs[i]

    # Get the original image, target mask, and prediction
    input_image = learn.dls.valid.dataset[idx][0]  # Fetches the input image
    target_mask = targs[idx].cpu()  # Moves targets to CPU
    pred_mask = preds[idx].argmax(dim=0).cpu()  # Get the most likely class for each pixel

    # Get the filename
    filename = highest_misses[i]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot original image
    show_img_or_tensor(input_image, ax=axs[0], title=f'Input Image:\n{filename}')
    # Plot target mask
    show_img_or_tensor(target_mask, mask_colors, codes, ax=axs[1], title='Target Mask')
    # Plot predicted mask
    show_img_or_tensor(pred_mask, mask_colors, codes, ax=axs[2], title='Predicted Mask')

    create_legend(mask_colors)
    plt.tight_layout()
    plt.show()

# %% calculate cleaned inf size
# Define the threshold for severe misses
threshold = 5  # Percent unit

# Correcting the data for predictions where the ground truth is 0 but prediction is above the threshold
correl_df.loc[(correl_df['infarct_size_per_AAR_Targets'] == 0) & 
              (correl_df['infarct_size_per_AAR_Predicted'] > threshold), 
              'infarct_size_per_AAR_Predicted'] = 0

# Create a new DataFrame from correl_df to keep original data unchanged if needed later
cleaned_df = correl_df.copy()

# Proceed with correlation plotting for all cleaned data
plot_correlation(cleaned_df, 'Cleaned All Data, Fold 1')

# Filter and plot for placebo only
placebo_data_clean = cleaned_df[cleaned_df['protocol'] == 'placebo']
plot_correlation(placebo_data_clean, 'Cleaned Placebo Data, Fold 1')

# Filter and plot for Protect only
protect_data_clean = cleaned_df[cleaned_df['protocol'] == 'Protect']
plot_correlation(protect_data_clean, 'Cleaned Protect Data, Fold 1')

# %%