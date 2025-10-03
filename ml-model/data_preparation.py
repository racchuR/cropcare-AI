"""
CropCare AI - Data Preparation Script
Utility script for preparing and organizing crop disease dataset
"""

import os
import shutil
import json
from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import random

def create_directory_structure(base_dir, class_names):
    """Create directory structure for dataset"""
    base_path = Path(base_dir)
    
    # Create main directories
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            (base_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in: {base_path}")

def organize_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Organize dataset into train/val/test splits"""
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Define class names
    class_names = [
        "Healthy",
        "Bacterial_Blight",
        "Brown_Spot",
        "Leaf_Blight",
        "Leaf_Scald",
        "Leaf_Spot",
        "Rust",
        "Smut"
    ]
    
    # Create target directory structure
    create_directory_structure(target_dir, class_names)
    
    total_images = 0
    class_counts = {split: {class_name: 0 for class_name in class_names} 
                   for split in ['train', 'val', 'test']}
    
    for class_name in class_names:
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Get all images for this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(list(class_dir.glob(ext)))
        
        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        total_class_images = len(image_files)
        train_count = int(total_class_images * train_ratio)
        val_count = int(total_class_images * val_ratio)
        
        # Split images
        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]
        
        # Copy images to respective directories
        for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img_path in images:
                target_img_path = target_path / split / class_name / img_path.name
                shutil.copy2(img_path, target_img_path)
                class_counts[split][class_name] += 1
                total_images += 1
        
        print(f"{class_name}: {total_class_images} images -> "
              f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Save dataset statistics
    stats = {
        'total_images': total_images,
        'class_counts': class_counts,
        'splits': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        }
    }
    
    with open(target_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset organization completed!")
    print(f"Total images: {total_images}")
    print(f"Statistics saved to: {target_path / 'dataset_stats.json'}")
    
    return stats

def validate_images(dataset_dir):
    """Validate images in dataset"""
    dataset_path = Path(dataset_dir)
    
    print("Validating images...")
    
    invalid_images = []
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
        
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    total_images += 1
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception as e:
                        invalid_images.append((str(img_path), str(e)))
    
    print(f"Validation completed!")
    print(f"Total images checked: {total_images}")
    print(f"Invalid images: {len(invalid_images)}")
    
    if invalid_images:
        print("\nInvalid images:")
        for img_path, error in invalid_images:
            print(f"  {img_path}: {error}")
    
    return invalid_images

def plot_dataset_distribution(dataset_dir):
    """Plot dataset distribution"""
    dataset_path = Path(dataset_dir)
    
    # Load statistics
    stats_file = dataset_path / 'dataset_stats.json'
    if not stats_file.exists():
        print("Dataset statistics not found. Run organize_dataset first.")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Prepare data for plotting
    splits = ['train', 'val', 'test']
    class_names = list(stats['class_counts']['train'].keys())
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, split in enumerate(splits):
        counts = [stats['class_counts'][split][class_name] for class_name in class_names]
        
        axes[i].bar(class_names, counts)
        axes[i].set_title(f'{split.capitalize()} Set')
        axes[i].set_xlabel('Disease Class')
        axes[i].set_ylabel('Number of Images')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(dataset_path / 'dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Distribution plot saved to: {dataset_path / 'dataset_distribution.png'}")

def create_sample_images(dataset_dir, output_dir, samples_per_class=5):
    """Create sample images for each class"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    class_names = [
        "Healthy", "Bacterial_Blight", "Brown_Spot", "Leaf_Blight",
        "Leaf_Scald", "Leaf_Spot", "Rust", "Smut"
    ]
    
    for class_name in class_names:
        class_dir = dataset_path / 'train' / class_name
        if not class_dir.exists():
            continue
        
        # Get sample images
        image_files = list(class_dir.glob('*'))
        sample_images = random.sample(image_files, min(samples_per_class, len(image_files)))
        
        # Create class output directory
        class_output_dir = output_path / class_name
        class_output_dir.mkdir(exist_ok=True)
        
        # Copy sample images
        for img_path in sample_images:
            target_path = class_output_dir / img_path.name
            shutil.copy2(img_path, target_path)
        
        print(f"Created {len(sample_images)} sample images for {class_name}")

def main():
    parser = argparse.ArgumentParser(description='Crop disease dataset preparation')
    parser.add_argument('--action', type=str, required=True, 
                       choices=['organize', 'validate', 'plot', 'samples'],
                       help='Action to perform')
    parser.add_argument('--source_dir', type=str, help='Source directory for organize action')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for samples action')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--samples_per_class', type=int, default=5, help='Samples per class')
    
    args = parser.parse_args()
    
    if args.action == 'organize':
        if not args.source_dir:
            print("Error: source_dir is required for organize action")
            return
        
        organize_dataset(args.source_dir, args.dataset_dir, 
                        args.train_ratio, args.val_ratio, args.test_ratio)
    
    elif args.action == 'validate':
        validate_images(args.dataset_dir)
    
    elif args.action == 'plot':
        plot_dataset_distribution(args.dataset_dir)
    
    elif args.action == 'samples':
        if not args.output_dir:
            print("Error: output_dir is required for samples action")
            return
        
        create_sample_images(args.dataset_dir, args.output_dir, args.samples_per_class)

if __name__ == "__main__":
    main()
