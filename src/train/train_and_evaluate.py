# %%
# import libraries
import pandas as pd
import numpy as np
import ast
import cv2
import math
from typing import List
import PIL
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score
from fastai.vision.all import *
from fastai.metrics import *
from scipy.ndimage.morphology import binary_erosion
from sklearn.metrics import f1_score
import labelbox
import urllib.request
import os
import json
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision import transforms
from torchvision.transforms import GaussianBlur

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
                    mask_cropped.save(os.path.join(output_dir, f"{base_name}_{mask_name}_L{file_suffix}.png"))

def merge_masks(masks, output_path, base_image_size):
    # DEBUG: print(f"Merging masks: {masks} into {output_path}")
    # Initialize the combined mask with zeros with correct dimensions
    combined_mask = np.zeros((base_image_size[1], base_image_size[0]), dtype=np.uint8)  # height first, then width
    for mask in masks:
        current_mask = Image.open(mask).convert('L')
        if current_mask.size != base_image_size:
            current_mask = current_mask.resize(base_image_size, Image.NEAREST)
        current_mask_array = np.array(current_mask)
        combined_mask = np.maximum(combined_mask, current_mask_array)
    combined_mask_image = Image.fromarray(combined_mask)
    combined_mask_image.save(output_path)

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
    file_base = original_file_name.replace(suffix + '.out.jpg', '')

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
            # print(f"Checked path not found: {mask_path}")  # Debug output for paths checked

    return pd.Series(filenames)

def get_filename(dl, idx):
    """Helper function to get filenames according to their index in a dataset."""
    fname = dl.dataset.items[idx]
    return os.path.basename(fname)

def get_transform(desired_size):
    """Return a transformation pipeline."""
    return T.Compose([
        T.Resize((desired_size, desired_size)),  # Resizes the image to the desired size
        T.ToTensor()  # Converts the image to a PyTorch tensor
    ])

def process_mask_from_row(row, mask_name, transform):
    mask_path = row[mask_name]
    if pd.notna(mask_path) and os.path.exists(mask_path):
        return transform(Image.open(mask_path)).to(device)[0]
    else:
        return torch.zeros((1, transform.transforms[0].size[0], transform.transforms[0].size[1]), device=device)

def process_mask_for_type(mask: torch.Tensor, mask_type: str, codes: List[str], desired_size: int = 384) -> torch.Tensor:
    """Extract a specific mask type from a combined mask image and resize."""
    if isinstance(mask, torch.Tensor) and mask.device.type == 'cuda':
        mask = mask.cpu()

    # Make sure to handle individual masks
    mask_array = np.array(mask.squeeze())  # Reduce dimensions if there are singleton dimensions

    # Ensure you're processing one mask at a time
    if mask_array.ndim == 3:  # This suggests (batch_size, height, width)
        raise ValueError("process_mask_for_type function is designed to handle one mask at a time.")
    
    mask_val = codes.index(mask_type)
    binary_mask = (mask_array == mask_val).astype(np.uint8)

    if binary_mask.ndim != 2:
        raise ValueError(f"Expected binary_mask to be 2D but got shape {binary_mask.shape}")
    
    binary_mask_img = Image.fromarray(binary_mask)  # Convert to PIL image

    # Define transformation
    transform = T.Compose([
        T.Resize((desired_size, desired_size)),
        T.ToTensor()
    ])
    resized_mask = transform(binary_mask_img)

    return resized_mask.float()

# Label function:
class LabelFunc:
    def __init__(self, mask_dir, mask_names):
        self.mask_dir = Path(mask_dir)
        self.mask_names = mask_names

    def __call__(self, fn):
        mask_base = fn.stem.replace('_cropped.out', '')
        combined_mask = None

        for i, mask_name in enumerate(self.mask_names):
            mask_file = self.mask_dir/f"{mask_base}_{mask_name}_L.png"
            if mask_file.exists():
                mask = Image.open(mask_file).convert('L')
                mask_array = np.array(mask)
                if combined_mask is None:
                    combined_mask = np.zeros(mask_array.shape, dtype=np.uint8)
                # print("Combined mask shape:", combined_mask.shape) # DEBUG!
                # print("Mask array shape:", mask_array.shape) # DEBUG!
                combined_mask[mask_array == 255] = i  # Ensure dimensions match

        if combined_mask is not None:
            return Image.fromarray(combined_mask, mode='L')
        else:
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
        inputs, targets = inputs.cuda(), targets.cuda()
        ce_loss = self.cross_entropy(inputs, targets)
        dice_loss = self.dice_loss(inputs, targets)
        inputs_softmax = self.softmax(inputs)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        l1_loss = self.l1_loss(inputs_softmax, targets_one_hot)
        boundary_loss = self.boundary_loss(inputs, targets)
        total_loss = ce_loss + self.dice_weight * dice_loss + self.l1_weight * l1_loss + self.boundary_weight * boundary_loss
        return total_loss

# data augmentation:
def custom_gaussian_blur(x, size: int = 5, sigma: float = 0.2):
    return GaussianBlur(kernel_size=size, sigma=sigma)(x)

# visualization:
class CustomSegmentationInterpretation:
    def __init__(self, learn, dl, preds, targs, losses):
        self.learn = learn
        self.dl = dl
        self.preds = preds
        self.targs = targs
        self.losses = losses

    @classmethod
    def from_learner(cls, learn, ds_idx=1, dl=None):
        if dl is None: dl = learn.dls[ds_idx].new(shuffle=False, drop_last=False)
        preds, targs = learn.get_preds(dl=dl, with_decoded=False)
        # Ensure preds and targs are on the same device as the model
        device = learn.model.parameters().__next__().device
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
    """Resizes a mask to match the target shape of an image."""
    # Here, ensure that the aspect ratio is preserved and nearest-neighbor interpolation is used
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
        mask_path = row[mask_name]
        if pd.notna(mask_path) and mask_path.endswith('.png'):
            try:
                mask = mpimg.imread(mask_path)
                if mask.ndim == 3 and mask.shape[2] == 4:
                    mask = mask[:, :, 0]  # Use only the first channel if it's a four-channel image.
                if mask.ndim == 2:  # Grayscale mask
                    mask = mask[:, :, None]  # Add a channel dimension.
                if mask.max() > 1:
                    mask = mask / 255.0
                
                mask = resize_mask_to_image(mask, image.shape[:2])

                colored_mask = np.zeros_like(image)
                for i in range(3):  # Apply color to RGB channels
                    colored_mask[:, :, i] = mask * color[i]

                overlayed_image += colored_mask
                overlayed_image = np.clip(overlayed_image, 0, 1)
            except FileNotFoundError:
                print(f"File not found for mask {mask_name}: {mask_path}")
        else:
            print(f"No mask file for {mask_name}")

    # Create legend handles manually
    legend_elements = [mpatches.Patch(facecolor=color[:3], edgecolor='r', label=name.replace('_', ' '))
                       for name, color in mask_colors.items() if pd.notna(row[name])]

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
    """Apply colors to a segmentation mask."""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    print("Color assignments:")  # Debugging output
    for idx, code in enumerate(codes):
        color = class_to_color[code]
        print(f"Code '{code}' ({idx}): {color}")  # Debugging output
        mask_class = mask == idx
        for c in range(3):  # Apply RGB channels
            colored_mask[:, :, c] += (mask_class * color[c]).astype(np.uint8)
    return colored_mask

def show_batch_with_legend(dls, codes, custom_colors, nrows=2, ncols=2, xb=None, yb=None):
    """Shows a batch of items in a dataloader with an added legend using custom colors."""
    if xb is None or yb is None:
        xb, yb = dls.one_batch()

    xb = xb.cpu()  # Move the batch of images to CPU
    yb = yb.cpu()  # Move the batch of masks to CPU

    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols*2, figsize=(12, 6))
    for i in range(nrows):
        for j in range(0, ncols*2, 2):
            # Display image
            axarr[i, j].imshow(xb[i * ncols + j // 2].permute(1, 2, 0))
            axarr[i, j].axis('off')
            
            # Display mask with explicit coloring
            mask = yb[i * ncols + j // 2].numpy()
            colored_mask = np.zeros((*mask.shape, 3))
            
            for class_idx, color_name in enumerate(codes):  # Class indices start at 0
                mask_class = mask == class_idx
                for k in range(3):  # For RGB channels
                    colored_mask[:, :, k] += mask_class * custom_colors[color_name][k]

            axarr[i, j+1].imshow(colored_mask)
            axarr[i, j+1].axis('off')

    # Add legend
    patches = [mpatches.Patch(color=custom_colors[code], label=code) for code in codes]
    fig.legend(handles=patches, loc='center', ncol=len(codes), bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.show()

def show_img_or_tensor(img_or_tensor, class_to_color=None, codes=None, ax=None, title=None):
    if ax is None:
        ax = plt.gca()

    if class_to_color is not None and codes is not None:
        # Convert the mask to its colored version if tensor or array
        if isinstance(img_or_tensor, torch.Tensor):
            img_or_tensor = img_or_tensor.detach().cpu().numpy()

        colored_mask = np.zeros((img_or_tensor.shape[0], img_or_tensor.shape[1], 3), dtype=float)
        for idx, code in enumerate(codes):
            mask = img_or_tensor == idx
            for channel in range(3):
                colored_mask[..., channel] += mask * class_to_color[code][channel]

        img_or_tensor = colored_mask

    if isinstance(img_or_tensor, torch.Tensor):
        img_or_tensor = img_or_tensor.permute(1, 2, 0).numpy()
    elif isinstance(img_or_tensor, PIL.Image.Image):
        img_or_tensor = np.array(img_or_tensor)

    ax.imshow(img_or_tensor)
    if title:
        ax.set_title(title)
    ax.axis('off')

def create_legend(class_to_color):
    """Create a custom legend using a class_to_color dictionary."""
    legend_patches = [Patch(color=color, label=label) for label, color in class_to_color.items()]
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def create_filename_to_index_mapping(dl):
    """Creates a mapping from filename to DataLoader index."""
    filename_to_index = {}
    for idx in range(len(dl.dataset.items)):
        filename = get_filename(dl, idx)
        filename_to_index[filename] = idx
    return filename_to_index

def get_indices_for_heart_id(df, heart_id, filename_to_index_mapping):
    """Returns a list of DataLoader indices for images associated with a specific heart ID."""
    filenames = df[df['heart_id'] == heart_id]['file_name'].tolist()
    indices = [filename_to_index_mapping.get(filename) for filename in filenames if filename in filename_to_index_mapping]
    return indices

# calculation of key features:
def calc_features_from_masks(infarct_mask, aar_mask, rv_mask, remote_mask):
    """Calculates segmentation target features"""
    infarct_area = torch.sum(infarct_mask > 0).item()
    aar_area = torch.sum(aar_mask > 0).item()
    rv_area = torch.sum(rv_mask > 0).item()
    remote_area = torch.sum(remote_mask > 0).item()

    total_aar = aar_area + infarct_area
    infarct_size_per_aar = (infarct_area / total_aar) * 100 if total_aar > 0 else 0
    lv_area = infarct_area + aar_area + remote_area

    return infarct_area, aar_area, rv_area, remote_area, infarct_size_per_aar, lv_area

def calculate_roundness(mask):
    """Calculate the roundness of single shapes from a mask area."""
    mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Scale mask to 0-255

    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze(0)  # Remove channel dimension

    # Threshold the mask to ensure it's binary
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roundness_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        roundness = 4 * math.pi * area / (perimeter**2) if perimeter > 0 else 0
        roundness_list.append(roundness)

    # print(f"Roundness calculated: {roundness_list}")  # Debug print
    return roundness_list

def calculate_overall_roundness(mask):
    """Calculate the roundness of the entire mask area as a single shape."""
    mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Scale mask to 0-255

    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze(0)  # Remove channel dimension

    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0  # No contours found

    all_contours = np.vstack(contours)
    area = cv2.contourArea(all_contours)
    perimeter = cv2.arcLength(all_contours, True)
    roundness = 4 * math.pi * area / (perimeter**2) if perimeter > 0 else 0
    return roundness

def count_partitions(mask):
    """Count the number of separate partitions (contours) in a mask."""
    mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # Scale mask to 0-255

    if mask_np.ndim > 2:
        mask_np = mask_np.squeeze(0)  # Remove channel dimension

    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    partitions = len(contours)
    # print(f"Partitions counted: {partitions}")  # Debug print
    return partitions

# dataloading
def splitter(items):
    """Custom splitter function based on preselected items in a dataframe"""
    train_idxs = []
    valid_idxs = []
    for i, img_path in enumerate(items):
        filename = img_path.name  # Get the filename
        row = df[df['file_name'] == filename]  # Match the filename in the DataFrame
        if not row.empty:
            if row['split'].values[0] == 'train':
                train_idxs.append(i)
            elif row['split'].values[0] == 'valid':
                valid_idxs.append(i)
    return train_idxs, valid_idxs

def get_image_files_func(dummy_arg=None):
    """Wrapper function to return image files and ensure picklable data structure"""
    return images

def get_y(img_path):
    filename = img_path.name
    mask = label_func(img_path)
    return mask

def calculate_class_weights(df, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPSILON = 1e-6  # Small number to prevent division by zero
    class_counts = torch.zeros(len(mask_names), device=device)

    for index, row in df.iterrows():
        for mask_name in mask_names:
            mask_path = row[mask_name]
            if pd.notna(mask_path) and os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask_tensor = transform(mask).to(device)[0]
                class_index = mask_names.index(mask_name)
                # Count pixels for the current class
                class_counts[class_index] += torch.sum(mask_tensor == class_index + 1)

    total_pixels = class_counts.sum()
    if total_pixels == 0:
        return np.zeros(len(mask_names))  # Avoid division by zero if no pixels found

    # Calculate the fractions of each class over the total pixel count
    fractions = class_counts / total_pixels

    # Invert fractions to get weights
    weights = 1.0 / (fractions + EPSILON)

    # Normalize weights so that their sum equals 1
    normalized_weights = weights / weights.sum()

    return normalized_weights.cpu().numpy()

# metrics calculations
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
    # Create binary masks for the specific class
    preds_i = (preds == class_idx).float()
    targs_i = (targs == class_idx).float()

    # Flatten the tensors
    preds_i = preds_i.view(-1)
    targs_i = targs_i.view(-1)

    # Compute the intersection and minimum value between sum of predictions and targets
    intersection = (preds_i * targs_i).sum()
    min_value = min(preds_i.sum(), targs_i.sum())

    # Compute Szymkiewicz-Simpson coefficient for this class
    overlap = (intersection + eps) / (min_value + eps)
    return overlap

def dice_score_single(preds, targs, class_idx, eps=1e-8):
    """Computes the Dice score for tensors from a single class.
    
    Args:
        preds (Tensor): A batch of predictions.
        targs (Tensor): A batch of targets.
        class_idx (int): The index of the respective classes.
        eps (float): Epsilon value to prevent division by zero.
        
    Returns:
        float: The average Dice score across all tensors from a class.
    """
    # Create binary masks for the specific class
    preds_i = (preds == class_idx).float()
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
    preds_i = (preds == class_idx).float()
    targs_i = (targs == class_idx).float()

    # Flatten the tensors
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
    # Convert predictions to probabilities for each class
    preds_prob = preds.softmax(dim=1)
    APs = []
    for i in range(num_classes):
        # Flatten the tensors using .reshape(...) as suggested
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

# GPU support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

## Step 0: download masks
# %% set up labelbox
# set up export function (export V2)
LB_API_KEY = ''
PROJECT_ID = ''
client = labelbox.Client(api_key = LB_API_KEY)
project = client.get_project(PROJECT_ID)
lb_export = project.export_v2(params={
	"data_row_details": True,
	"metadata_fields": True,
	"attachments": True,
	"project_details": True,
	"performance_details": True,
	"label_details": True,
	"interpolated_frames": True
  })

# lb_labels = lb_export.result

# %% manual download, loading of a ndjson file
ndjson_path = ''

# Initialize a list
lb_labels = []

# Open and read the NDJSON file line by line
with open(ndjson_path, 'r') as file:
    for line in file:
        # Convert each JSON line to a dictionary and append to the list
        lb_labels.append(json.loads(line))

# Now, `records` contains all the data as a list of dictionaries.
print(f"Loaded {len(lb_labels)} records from the NDJSON file.")

# %% explore export results
# Print the structure of the first item in lb_labels
print("Structure of an item in lb_labels:")
print(lb_labels[0])

# Print the keys of the first item
print("\nKeys of the first item:")
print(lb_labels[0].keys())

# Check and print the annotations part
print("\nExample of annotations in the first item:")
annotations = lb_labels[0]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']
print(annotations)

# %% class label and bbox download
# !!!Just download once!!!
# Create a dictionary to hold the mapping from filenames to mask URLs and bounding box data
mapping = {}

# Iterate through the labels
for item in lb_labels:
    # Extract the filename (external_id)
    filename = item['data_row']['external_id']
    
    # Extract annotations
    annotations = item['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']
    annotations_dict = {}

    for annotation in annotations:
        if annotation['annotation_kind'] == 'ImageSegmentationMask':
            # Handle mask annotations
            mask_name = annotation['name'].replace(' ', '_') + '_L' # mark labels with "L"
            mask_url = annotation['mask']['url']
            annotations_dict[mask_name] = mask_url
        elif annotation['annotation_kind'] == 'ImageBoundingBox':
            # Handle bounding box annotations
            bbox_name = annotation['name'].replace(' ', '_') + '_bbox' # mark bounding boxes
            bbox_data = annotation['bounding_box']  # Extract bounding box data
            annotations_dict[bbox_name] = bbox_data

    # Store the information in the dictionary
    mapping[filename] = annotations_dict

# Create a directory to store the masks and bounding boxes
os.makedirs('masks', exist_ok=True)

# Define headers for authentication with Labelbox
headers = {'Authorization': f'Bearer {LB_API_KEY}'}

# Iterate through the mapping and process the masks and bounding boxes
for filename, annotations in mapping.items():
    filename_without_extension = os.path.splitext(filename)[0].replace('.out', '')
    
    for annotation_name, annotation_data in annotations.items():
        if annotation_name.endswith('_L'):
            # Process masks
            mask_url = annotation_data
            mask_filename = os.path.join('masks', filename_without_extension + '_' + annotation_name + '.png')
            req = urllib.request.Request(mask_url, headers=headers)
            try:
                image = Image.open(urllib.request.urlopen(req))
                image.save(mask_filename)
            except Exception as e:
                print(f"An error occurred while processing {mask_filename}: {str(e)}")
        elif annotation_name.endswith('_bbox'):
            # Process bounding boxes
            bbox_filename = os.path.join('masks', filename_without_extension + '_' + annotation_name + '.txt')
            with open(bbox_filename, 'w') as file:
                file.write(str(annotation_data))  # Convert bounding box data to string and save

# %% class label download, check for missing
# check for missing files, generate expected filenames list:
# This script assumes that the 'mapping' dictionary is already filled from previous run

expected_filenames = []

# Iterate through the mapping to generate the expected filenames
for filename, masks in mapping.items():
    # Remove the original extension and ".out" from the filename
    filename_without_extension = os.path.splitext(filename)[0].replace('.out', '')
    
    for mask_name, mask_url in masks.items():
        # Replace spaces with underscores in the mask_name
        mask_name = mask_name.replace(' ', '_')
        
        # Create the filename for the mask
        expected_filename = filename_without_extension + '_' + mask_name + '.png'
        expected_filenames.append(expected_filename)

# Now, compare this list with the files that are actually present in the 'masks' directory

# List all files in your 'masks' directory
downloaded_files = set(os.listdir('masks'))

# Convert your list of expected filenames to a set
expected_files_set = set(expected_filenames)

# Find out which files were expected but are missing
missing_files = expected_files_set - downloaded_files

print(f"Missing files: {missing_files}")

# %% class label download, get missing
# get missing files:
# List of failed filenames
# List of specific filenames to re-download
missing_filenames = [
    ""
    # Add more filenames as needed
]

# Define headers for authentication with Labelbox
headers = {'Authorization': f'Bearer {LB_API_KEY}'}

# Create a set for faster lookup
specific_filenames_set = set(missing_filenames)

# Iterate through the mapping and download only specific masks
for filename, annotations in mapping.items():
    # Remove the original extension and ".out" from the filename
    filename_without_extension = os.path.splitext(filename)[0].replace('.out', '')
    
    for annotation_name, annotation_data in annotations.items():
        # Construct the mask filename
        mask_filename = filename_without_extension + '_' + annotation_name.replace(' ', '_') + '.png'

        # Check if the mask is in the specific list
        if mask_filename in specific_filenames_set:
            # Process masks
            mask_url = annotation_data
            full_mask_filename = os.path.join('masks', mask_filename)

            # Make the API request with the defined headers
            req = urllib.request.Request(mask_url, headers=headers)

            # Download and save the mask
            try:
                image = Image.open(urllib.request.urlopen(req))
                image.save(full_mask_filename)
                print(f"Successfully redownloaded {full_mask_filename}")
            except Exception as e:
                print(f"An error occurred while processing {full_mask_filename}: {str(e)}")

# %% check downlaoded labels
# check the number of downloaded masks:
print(f"Number of downloaded mask files: {len(os.listdir('masks'))}")

# check for downloaded masks as pngs
mask_files = os.listdir('masks')
for file in mask_files:
    print(file)

# Example usage: Get the masks for a example image (im)
im = 'Protect_V37D17_6_a_IMG_4278.out.jpg'
# Extracting the base name to search for corresponding masks
base_name = im.split('.')[0]

# Finding all the masks related to the base_name
masks_im = [file for file in mask_files if base_name in file]

print(masks_im)

# Display the first 5 mask images
for file in mask_files[:5]:
    img_path = os.path.join('masks', file)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(file)
    plt.show()

# %% step 0.5 Preprocessing
# crop images and labels to their respective 
mask_names = ['infarct', 'lumen', 'epicardium', 'right_ventricle', 'remote_areal', 
              'hemorrhage', 'area_at_risk_without_infarct,_hemorrhage', 'remaining_areas']

im_dir = ''
mask_dir = ''
out_dir = ''

crop_files(im_dir, mask_dir, mask_names, out_dir)

# %% move and combine masks to a new directory for a reduced parameter model
# Define paths
src_folder = ''  # Source folder where the original images and masks are stored
dest_folder = ''  # Destination folder to store combined and renamed masks

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

for root, _, files in os.walk(src_folder):
    grouped_files = {}
    for file in files:
        parts = file.split('_')
        if len(parts) > 5:
            base_name = '_'.join(parts[:5])
            grouped_files.setdefault(base_name, []).append(file)

    for base_name, group_files in grouped_files.items():
        aar_masks = []
        remaining_masks = []
        output_paths = {}
        base_image_size = None

        for file_name in group_files:
            full_path = os.path.join(root, file_name)
            if file_name.endswith('_cropped.out.jpg'):
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

# %% check for mismatched dimensions
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
file_name = 'TTC-datasets.xlsx'
# Load base data without split information
df = pd.read_excel(path + '/' + file_name, header=None, usecols=range(5))
df.columns = ['file_name', 'label', 'heart_id', 'slice_number', 'slice_side']

# Load the specific column for the current fold based on its index
fold = 0  # Adjust this for each fold (0 for first fold, 1 for second, and so on)
fold_idx = fold + 5  # Adjust the index to match the column in the Excel file
split_column = pd.read_excel(path + '/' + file_name, header=None, usecols=[fold_idx])

# Since read_excel returns a DataFrame, extract the column as a Series
split_column = split_column.iloc[:, 0]

# Add the split column to the DataFrame
df['split'] = split_column

# construct dataframe with masks
# add path for masks to the dataframe
# Caution: Just run once!
# Your original DataFrame 'df' and other setup code
mask_names = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']
mask_dir = path

# Define the suffix used in the file renaming
suffix = '_cropped'

# Update the file_name column to include the _cropped suffix for images
df['file_name'] = df['file_name'].str.replace('.out.jpg', f'{suffix}.out.jpg')

# Apply the updated function to generate mask filenames
mask_filenames_df = df.apply(lambda row: get_mask_filenames(row, mask_dir, mask_names, suffix), axis=1)

# Concatenate the mask filenames with the original DataFrame
df = pd.concat([df, mask_filenames_df], axis=1)

print(df.head())

# %% Step 1.5 Visulization of base images and masks
# attach masks to original images for visualization:
# Create a dictionary of colors, skipping the first color in the palette
mask_colors = {
    'infarct': (1, 1, 1),  # white
    'AAR': (1, 0.5, 0.5),  # light red
    'remote': (1, 0, 0),  # red
    'right_ventricle': (0, 0, 1),  # blue
    'remaining': (0, 0, 0)  # black
    }

# Example usage
image_filename = 'Protect_V14D21_4_a_IMG7657_cropped.out.jpg'
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
# apply size transformation from training segment
DESIRED_SIZE = 384
transform = get_transform(DESIRED_SIZE)

# Iteration through DataFrame to calculate target features
results = []

for index, row in df.iterrows():
    # Process masks for each category
    infarct_mask = process_mask_from_row(row, 'infarct', transform)
    aar_mask = process_mask_from_row(row, 'AAR', transform)
    rv_mask = process_mask_from_row(row, 'right_ventricle', transform)
    remote_mask = process_mask_from_row(row, 'remote', transform)

    # Calculate features
    infarct_area, aar_total, rv_area, remote_area, infarct_size_per_aar, lv_area = calc_features_from_masks(infarct_mask, aar_mask, rv_mask, remote_mask)
    roundness_single = calculate_roundness(infarct_mask)
    roundness_av = np.mean(roundness_single) if roundness_single else 0
    roundness_overall = calculate_overall_roundness(infarct_mask)
    partitions = count_partitions(infarct_mask)

    # Append results
    results.append([row['file_name'], row['label'], infarct_area, aar_total, rv_area, remote_area, infarct_size_per_aar, lv_area, partitions, roundness_single, roundness_av, roundness_overall])

# Define columns for DataFrame
columns = ['filename', 'protocol', 'infarct_area', 'aar_total', 'rv_area', 'remote_area', 'infarct_size_per_aar', 'lv_area', 'partitions', 'roundness_single', 'roundness_av', 'roundness_overall']
results_df = pd.DataFrame(results, columns=columns)

# Displaying the results
print(results_df)

# %% dataframe and saving
# Save the results DataFrame to an Excel file
results_df.to_excel('traget features segmentation.xlsx', index=True)

# %% Visualize the target features
# Setting the style
sns.set_style("whitegrid")

# List of primary metrics to be plotted
primary_metrics = ['infarct_area', 'aar_total', 'infarct_size_per_aar', 'lv_area', 'rv_area', 'remote_area']

# Plotting each primary metric
for metric in primary_metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='protocol', y=metric, palette="colorblind")
    plt.title(f'Boxplot of {metric} by Protocol')
    plt.xlabel('Protocol')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.show()

# Additional metrics related to shape and structure
additional_metrics = ['roundness_av', 'roundness_overall', 'partitions']

# Plotting each additional metric
for metric in additional_metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='protocol', y=metric, palette="colorblind")
    plt.title(f'Boxplot of {metric} by Protocol')
    plt.xlabel('Protocol')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.show()


# %% Step 3: train a image segmentation model:
## Dataloading:
# path to the data, the ground truth original images, class labels and bounding boxes
data = Path(path)
images = get_image_files(data)

# define class names ("codes") for the segmentation classes
mask_names = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']

# Initialize the label function with the path and mask names
label_func = LabelFunc(data, mask_names)

SIZE = 384

transforms = [
    RandomResizedCrop(SIZE, min_scale=0.7, ratio=(1, 1)),
    FlipItem(p=0.5),
    Rotate(max_deg=180, p=0.5, pad_mode='zeros'),
    Dihedral(p=0.5, pad_mode='zeros'),
    Brightness(max_lighting=0.2, p=0.75),
    Contrast(max_lighting=0.4, p=0.75),
    Saturation(max_lighting=0.2, p=0.75),
    Warp(magnitude=0.4, p=0.2, pad_mode='zeros'),
    Zoom(max_zoom=1.5, min_zoom=0.75, p=0.5),
    RandomErasing(p=0.2, sl=0.1, sh=0.2, max_count=3),
    custom_gaussian_blur,
    # Normalize.from_stats(mean=[0.3153, 0.1492, 0.1615], std=[0.2348, 0.1886, 0.1993])
    ]

# DataBlock using get_image_files_func, splitter and label_func custom functions
dblock = DataBlock(
    blocks=(ImageBlock, MaskBlock(mask_names)),
    splitter=splitter,
    get_items=get_image_files_func,
    get_y=label_func,
    item_tfms=Resize(SIZE, method=ResizeMethod.Pad, pad_mode='zeros'),
    batch_tfms=transforms
)

# Create DataLoaders
dls = dblock.dataloaders(data, bs = 26)

# %% check loaded data (1)
# check loaded data
dls.show_batch(unique=False, max_n=8) # unique=True for augmentations on one image

# %% check loaded data (2)
# show batch with legend to be sure that all the labeles are correctly applied
# Define a custom colormap
mask_colors = {
    'remaining': (0, 0, 0),  # black
    'infarct': (1, 1, 1),  # white
    'AAR': (1, 0.7, 0.7),  # light red
    'remote': (1, 0, 0),  # red
    'right_ventricle': (0, 0, 1),  # blue
}

# use the costum show_batch function with the color dictionary
show_batch_with_legend(dls, mask_names, mask_colors, nrows=2, ncols=2)

# %% check loaded data (3)
# %% Class weights:
# calculate class weights to handle class imbalances in small areas
mask_names = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']
df = df
SIZE = 384
transform = get_transform(SIZE)

class_weights = calculate_class_weights(df, transform)
print("Normalized Class Weights:", class_weights)

# %% define model parameters
# load last calculated normalized class weights
normalized_weights = [0.1, 0.6, 0.4, 0.1, 0.2]

# build the learner
dice = DiceMulti() # define metrics, here Dice coefficient

# specifiy class weights for the Loss-function
weights_tensor = torch.tensor(normalized_weights).cuda()
criterion = nn.CrossEntropyLoss(weight=weights_tensor).cuda()
custom_loss = CombinedCustomLoss(weights=normalized_weights).cuda()

# create the learner for dynamic U-Net
learn = unet_learner(dls, resnet34, 
                     metrics=[foreground_acc, dice],
                     opt_func=Adam, 
                     loss_func=custom_loss)

# %% print model
print(learn.model)

# %%
# empty cache
torch.cuda.empty_cache()

# %% learning rate
# get the learning rate
lr = learn.lr_find() 
lrv = lr.valley
print(lr)

# %% train
# train the model
learn.fit_one_cycle(300, lrv, wd=0.05, 
                    cbs=[ReduceLROnPlateau(monitor='valid_loss', factor=0.5, patience=2),
                         SaveModelCallback(monitor='dice_multi', every_epoch=True)])

# %% results
# show results:
learn.show_results(max_n=6)

# %% plot loss
# store loss and metrics printed during training (recorder somehow doesn't work here)
training_stats = ''

stats_df = pd.read_csv(training_stats, sep='\t')  # Assuming the data is tab-separated

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(stats_df['epoch'], stats_df['train_loss'], label='Train Loss', color=color)
ax1.plot(stats_df['epoch'], stats_df['valid_loss'], label='Validation Loss', color='tab:orange')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')
ax1.grid(True)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Dice Score', color=color)  # we already handled the x-label with ax1
ax2.plot(stats_df['epoch'], stats_df['dice_multi'], label='Dice Score', color=color)
ax2.plot(stats_df['epoch'], stats_df['foreground_acc'], label='pixel accuracy', color='tab:green')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 1])  # Adjusting the y-axis for dice score range
ax2.legend(loc='right')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Training / Validation Loss and Dice Score')
plt.show()

# %% find best model
# normalize metrics
scaler = MinMaxScaler()

# Normalize the metrics
stats_df[['valid_loss_norm', 'dice_multi_norm']] = scaler.fit_transform(stats_df[['valid_loss', 'dice_multi']])

# Inverse Dice score normalization to make higher scores better
stats_df['dice_multi_norm_inv'] = 1 - stats_df['dice_multi_norm']

# Calculating a combined score with normalization
stats_df['combined_score_norm'] = stats_df['dice_multi_norm'] / stats_df['valid_loss_norm']

# With normalization
best_epochs = stats_df.sort_values(by='combined_score_norm', ascending=False).head(3)
print(best_epochs)

# %% load best meodel
# load best model
best_epoch = best_epochs.iloc[0]['epoch']  
best_epoch = 289 # manual selection

# Load the model (adjust the path and model name as per your setup)
models_path = ''
best_model= f'{models_path}/model_{best_epoch}'
learn.load(best_model)

# %% save
# Save the model
model_fn = ''
# learn.save(model_fn) # save just the weights
learn.export(f'{model_fn}.pkl') # save including model architecture, dls, ...

# %% Load the model
# load model saved a .pkl
learn = load_learner('')

# %%
# predict with model
learn.predict('')

# %%
# load model saved as .pth (recreate dataloaders and learner first)
model_fn = ''
learn.load(f'/{model_fn}')

# %% Step 4: calculate metrics:
# get predictions and targets
preds, targs = learn.get_preds(dl=dls.valid)

# %% metrics per class
# calculate Dice- and Jaccard-Scores per class in the segmentation
# List of class names
classes = ['remaining', 'infarct', 'AAR', 'remote', 'right_ventricle']

# Convert raw predictions to class indices
preds_classes = preds.argmax(dim=1)
num_classes = 5

# Convert to class indices and then to numpy array
preds_np = preds.argmax(dim=1).cpu().numpy()  
targs_np = targs.cpu().numpy()  # Assuming targs is already (N, H, W)

# Calculate boundary F1 scores for all classes
boundary_f1_scores = boundary_f1_score(preds_np, targs_np, num_classes)

# Compute other scores and average precision for each class
scores = []
APs = average_precision_single(preds, targs, num_classes)

for i, class_name in enumerate(mask_names):
    dice_score = dice_score_single(preds_classes, targs, i).item()
    jaccard_score = jaccard_score_single(preds_classes, targs, i).item()
    overlap = simple_overlap(preds_classes, targs, i).item()
    ap = APs[i]
    # Append class scores along with the boundary F1 score for this class
    scores.append([class_name, overlap, dice_score, jaccard_score, ap, boundary_f1_scores[i]])

# Create DataFrame with updated columns including Boundary F1 scores
scores_df = pd.DataFrame(scores, columns=['Class', 'Overlap', 'Dice coeff', 'Jaccard coeff', 'AP', 'Boundary F1'])

# Calculate pixel accuracy for each class
class_accuracies = [class_pixel_accuracy(preds_np, targs_np, i) for i in range(num_classes)]

# Add pixel accuracy to the scores dataframe
scores_df['Pixel Accuracy'] = class_accuracies

print(scores_df)

# %% metrics overall
# calculate overall overlap and Dice-/Jaccard-Score for multiclass segmentation
# Convert raw predictions to class indices
preds_classes = preds.argmax(dim=1)

# calculate the similarity scores
overall_overlap = overlap_multi(preds_classes, targs, len(mask_names)).item()
dice_score_value = dice_score_multi(preds_classes, targs, num_classes=num_classes).item()  
jaccard_score_value = jaccard_score_multi(preds_classes, targs, num_classes=num_classes).item()
weighted_acc = weighted_pixel_accuracy(preds_classes, targs, len(mask_names))
APs = []
for i in range(num_classes):
    # Reshape the tensors to 1D
    preds_i = preds[:, i].reshape(-1)
    targs_i = (targs == i).long().reshape(-1)
    # Calculate average precision for this class
    AP = average_precision_score(targs_i, preds_i)
    APs.append(AP)

mAP = mean_average_precision(APs)
overall_boundary_f1 = sum(boundary_f1_scores) / len(boundary_f1_scores)

print('Weighted Pixel Accuracy:', weighted_acc)
print('Overall overlap:', overall_overlap)
print('Dice score: ', dice_score_value)
print('Jaccard score: ', jaccard_score_value)
print('Mean Average Precision:', mAP)
print('Overall Boundary F1 score:', overall_boundary_f1)

# %% Step 5: Interpretation and calculation of key features
# get interpretation data as an object
interp = CustomSegmentationInterpretation.from_learner(learn)

# %%
# define masks colors like before
mask_colors_RGB = {
    'remaining': (0, 0, 0),          # Black
    'infarct': (1, 1, 1),            # White
    'AAR': (1, 0.5, 0.5),            # Light Red
    'remote': (1, 0, 0),             # Red
    'right_ventricle': (0, 0, 1)     # Blue
}

# Correct the colors by ensuring they are within 0-1 range for Matplotlib
mask_colors = {k: tuple(c/255 if max(v) > 1 else c for c in v) for k, v in mask_colors_RGB.items()}

# %% check individual images from the valid dataset
idx = 100   # get an image from the valid dataset 

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
show_img_or_tensor(target_mask, mask_colors, mask_names, ax=axs[1], title='Target Mask')
# Plot predicted mask
show_img_or_tensor(pred_mask, mask_colors, mask_names, ax=axs[2], title='Predicted Mask')

# Create a legend for the masks
create_legend(mask_colors)

plt.tight_layout()
plt.show()

# %%
# visualize all images for specific ID
# Use the function to create the mapping for the validation DataLoader
filename_to_index_mapping = create_filename_to_index_mapping(learn.dls.valid)

# Example usage:
heart_id = "V04D20"  # Specify the heart ID you're interested in
indices = get_indices_for_heart_id(df, heart_id, filename_to_index_mapping)

# Visualize the results for each index
for idx in indices:
    input_image = learn.dls.valid.dataset[idx][0]
    target_mask = targs[idx].cpu()
    pred_mask = preds[idx].argmax(dim=0).cpu()
    filename = get_filename(learn.dls.valid, idx)

    # Plotting setup
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    show_img_or_tensor(input_image, ax=axs[0], title=f'Input Image:\n{filename}')
    show_img_or_tensor(target_mask, mask_colors, mask_names, ax=axs[1], title='Target Mask')
    show_img_or_tensor(pred_mask, mask_colors, mask_names, ax=axs[2], title='Predicted Mask')
    plt.tight_layout()
    plt.show()

# %% Top losses
# Get the top k losses and their indices
k = 20
top_losses, top_idxs = interp.top_losses(k=k)

# Iterate over the top losses
for i in range(k):
    idx = top_idxs[i]
    actual = targs[idx]  
    pred = preds[idx]
    loss = top_losses[i]

    filename = get_filename(learn.dls.valid, idx)

    print(f"Top {i+1} Loss: {loss.item()}, {filename}")

## Visualize the top losses
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
    show_img_or_tensor(target_mask, mask_colors, mask_names, ax=axs[1], title='Target Mask')
    # Plot predicted mask
    show_img_or_tensor(pred_mask, mask_colors, mask_names, ax=axs[2], title='Predicted Mask')

    # Create a legend for the masks
    create_legend(mask_colors)

    plt.tight_layout()
    plt.show()

# %% Lowest Losses
# Get the lowest k losses and their indices
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
    show_img_or_tensor(target_mask, mask_colors, mask_names, ax=axs[1], title='Target Mask')
    # Plot predicted mask
    show_img_or_tensor(pred_mask, mask_colors, mask_names, ax=axs[2], title='Predicted Mask')

    # Create a legend for the masks
    create_legend(mask_colors)

    plt.tight_layout()
    plt.show()

# %% Calculate key features from the valid dataset (1. ground truth, 2. predictions)
# Calculate features from ground truth in the valid dataset
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
# Define dataframe
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
with pd.ExcelWriter('valid set key features_fold5.xlsx') as writer:
    targets_df.to_excel(writer, sheet_name='Targets', index=False)
    preds_df.to_excel(writer, sheet_name='Predictions', index=False)

# %%
# Add protocol to both dataframes
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
# visualize data via scatter plot and correlation
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
# re-calculate correlation without highest misses
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

# %% combine slicewise predictions to infarct size [%AAR] per heart
# Apply cleaning rule to validation dataframe
# targets_df.loc[(targets_df['infarct_size_per_aar'] == 0) & (targets_df['infarct_size_per_aar'] > threshold), 'infarct_size_per_aar'] = 0

# Apply cleaning rule to predictions dataframe
# preds_df.loc[(preds_df['infarct_size_per_aar'] == 0) & (preds_df['infarct_size_per_aar'] > threshold), 'infarct_size_per_aar'] = 0

# Step 1: Extract Experiment IDs
targets_df['experiment_id'] = targets_df['filename'].apply(lambda x: x[8:14])
preds_df['experiment_id'] = preds_df['filename'].apply(lambda x: x[8:14])

# Step 2: Aggregate data by experiment ID and protocol
def aggregate_data(df):
    agg_df = df.groupby(['experiment_id', 'protocol']).agg(
        sum_infarct=('infarct_area', 'sum'),
        sum_aar=('aar_area', 'sum')
    ).reset_index()
    # Calculate total infarct size per AAR
    agg_df['total_infarct_size_per_aar'] = (agg_df['sum_infarct'] / agg_df['sum_aar']) * 100  # Percentage calculation
    return agg_df

targets_agg_df = aggregate_data(targets_df)
preds_agg_df = aggregate_data(preds_df)

# Merge the two aggregated DataFrames
agg_df = pd.merge(targets_agg_df, preds_agg_df, on=['experiment_id', 'protocol'], suffixes=('_Targets', '_Predicted'))

# Step 4: Plotting correlation
def plot_agg_correlation(data, x_axis='total_infarct_size_per_aar_Targets', y_axis='total_infarct_size_per_aar_Predicted', title='Correlation Analysis'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_axis, y=y_axis, data=data)
    plt.xlim(0, max(data[x_axis].max(), data[y_axis].max()) + 5)
    plt.ylim(0, max(data[x_axis].max(), data[y_axis].max()) + 5)

    r, _ = pearsonr(data[x_axis].dropna(), data[y_axis].dropna())
    mae = mean_absolute_error(data[x_axis], data[y_axis])
    rmse = np.sqrt(mean_squared_error(data[x_axis], data[y_axis]))

    max_value = max(data[x_axis].max(), data[y_axis].max())
    plt.plot([0, max_value], [0, max_value], 'k--', label='Line of Identity')
    m, b = np.polyfit(data[x_axis], data[y_axis], 1)
    plt.plot(data[x_axis], m * data[x_axis] + b, 'r--', label=f'Correlation Line (r={r:.2f})')

    plt.xlabel('Targets - Infarct Size [%AAR]')
    plt.ylabel('Predicted Infarct Size [%AAR]')
    plt.title(title)
    plt.legend(title=f'MAE: {mae:.2f}, RMSE: {rmse:.2f}')
    plt.show()

# Plot correlation for all data
plot_agg_correlation(agg_df, title='Overall Correlation of Infarct Size per AAR')

# Plot for placebo and Protect separately
placebo_agg_df = agg_df[agg_df['protocol'] == 'placebo']
plot_agg_correlation(placebo_agg_df, title='Placebo Correlation of Infarct Size per AAR')
protect_agg_df = agg_df[agg_df['protocol'] == 'Protect']
plot_agg_correlation(protect_agg_df, title='Protect Correlation of Infarct Size per AAR')
# %%

