# -*- coding: utf-8 -*-
"""
Created on Thu May 29 03:13:18 2025

@author: tak20
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

def create_black_mask_from_image(image):
    if len(image.shape) == 3: 
        black_pixels = (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)
    else: 
        black_pixels = (image == 0)
    
    return black_pixels

def compare_images_fill_black(original_path, target_paths, output_dir='./output', target_size=(1050, 864)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        original_img = cv2.imread(original_path)
        if original_img is None:
            raise FileNotFoundError(f"Original image not found: {original_path}")
        
        original_img_resized = cv2.resize(original_img, target_size, interpolation=cv2.INTER_AREA)
        original_img_gray = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2GRAY)
        
        original_processed_path = os.path.join(output_dir, 'original_resized.jpg')
        cv2.imwrite(original_processed_path, original_img_resized)
        print(f"Original image '{original_path}' resized to {target_size[0]}x{target_size[1]} and saved: {original_processed_path}")

    except Exception as e:
        print(f"Error loading or processing original image: {e}")
        return

    image_names = []
    diff_percentages = []

    print(f"\nComparing with original image: {original_path}")

    for target_path in target_paths:
        try:
            target_img = cv2.imread(target_path)
            if target_img is None:
                print(f"Warning: Comparison image not found: {target_path} - Skipping.")
                continue

            target_img_resized = cv2.resize(target_img, target_size, interpolation=cv2.INTER_AREA)
            
            target_img_for_comparison = target_img_resized.copy()
            
            black_mask_in_target = create_black_mask_from_image(target_img_resized)
            
            target_img_for_comparison[black_mask_in_target] = original_img_resized[black_mask_in_target]
            
            target_img_gray_filled = cv2.cvtColor(target_img_for_comparison, cv2.COLOR_BGR2GRAY)

            base_target_name = os.path.basename(target_path).split('.')[0]
            filled_target_path = os.path.join(output_dir, f'{base_target_name}_filled_from_original.jpg')
            
            cv2.imwrite(filled_target_path, target_img_for_comparison)
            
            print(f"  Comparison image '{target_path}' resized to {target_size[0]}x{target_size[1]}.")
            print(f"  Black areas filled with original image content and saved: {filled_target_path}")

            s = ssim(original_img_gray, target_img_gray_filled, 
                     data_range=target_img_gray_filled.max() - target_img_gray_filled.min())

            diff_percentages.append((1 - s) * 100)

            image_names.append(base_target_name)
            print(f"    SSIM (Similarity): {s:.4f}")
            print(f"    Difference (Absolute, black areas filled from original): {diff_percentages[-1]:.2f}%")

        except Exception as e:
            print(f"Error processing comparison image '{target_path}': {e} - Skipping.")
            continue

    if not image_names:
        print("No comparison images were processed.")
        return

    x = np.arange(len(image_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects = ax.bar(x, diff_percentages, width, label='Difference (%)', color='skyblue')
    ax.set_ylabel('Difference (%)', color='blue')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title(f'Image Comparison Results (Black areas filled from original) - All images resized to {target_size[0]}x{target_size[1]}')
    ax.set_xticks(x)
    ax.set_xticklabels(image_names, rotation=45, ha="right")
    ax.legend(loc='upper left')

    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    fig.tight_layout()

    output_graph_path = os.path.join(output_dir, 'comparison_graph_filled_black.png')
    plt.savefig(output_graph_path)
    print(f"\nGraph saved to: {output_graph_path}")

    plt.show()

if __name__ == "__main__":
    original_image_path = './image/original.jpg'
    target_image_paths = [
        './output/similarity.jpg',
        './output/affine.jpg',
        './output/perspective.jpg'
    ]
    output_directory = './output'
    unified_size = (1050, 864)

    compare_images_fill_black(original_image_path, target_image_paths, output_directory, unified_size)