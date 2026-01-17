"""
Create a subset of COCO 2017 validation dataset
Extracts 500 images with their corresponding annotations
"""
import json
import shutil
import random
from pathlib import Path

def create_coco_subset(
    source_images_dir="data/val2017/val2017",
    source_annotations="data/annotations_trainval2017/annotations/instances_val2017.json",
    output_dir="data/coco_subset",
    num_images=500,
    seed=42
):
    """
    Create a subset of COCO dataset with specified number of images
    
    Args:
        source_images_dir: Path to source images
        source_annotations: Path to source annotations JSON
        output_dir: Output directory for subset
        num_images: Number of images to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    output_path = Path(output_dir)
    images_out = output_path / "images"
    annotations_out = output_path / "annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    annotations_out.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from {source_annotations}...")
    with open(source_annotations, 'r') as f:
        coco_data = json.load(f)
    
    # Get all available images
    all_images = coco_data['images']
    print(f"Total images in dataset: {len(all_images)}")
    
    # Randomly select subset
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    selected_image_ids = {img['id'] for img in selected_images}
    
    print(f"Selected {len(selected_images)} images")
    
    # Copy selected images
    source_img_path = Path(source_images_dir)
    copied_count = 0
    
    print("Copying images...")
    for img_info in selected_images:
        src_file = source_img_path / img_info['file_name']
        dst_file = images_out / img_info['file_name']
        
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            copied_count += 1
            if copied_count % 100 == 0:
                print(f"  Copied {copied_count}/{len(selected_images)} images")
        else:
            print(f"  Warning: {src_file} not found")
    
    print(f"Copied {copied_count} images")
    
    # Filter annotations for selected images
    print("Filtering annotations...")
    filtered_annotations = [
        ann for ann in coco_data['annotations'] 
        if ann['image_id'] in selected_image_ids
    ]
    
    # Create new annotation file
    subset_coco = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': selected_images,
        'annotations': filtered_annotations,
        'categories': coco_data['categories']
    }
    
    # Save subset annotations
    output_annotation_file = annotations_out / "instances_val2017_subset.json"
    print(f"Saving annotations to {output_annotation_file}...")
    with open(output_annotation_file, 'w') as f:
        json.dump(subset_coco, f)
    
    # Create summary file
    summary_file = output_path / "subset_info.txt"
    with open(summary_file, 'w') as f:
        f.write(f"COCO 2017 Validation Subset\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Number of images: {len(selected_images)}\n")
        f.write(f"Number of annotations: {len(filtered_annotations)}\n")
        f.write(f"Number of categories: {len(coco_data['categories'])}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"\nImage directory: {images_out}\n")
        f.write(f"Annotation file: {output_annotation_file}\n")
        f.write(f"\nCategories:\n")
        for cat in coco_data['categories']:
            f.write(f"  {cat['id']}: {cat['name']}\n")
    
    print(f"\n✅ Subset created successfully!")
    print(f"📁 Images: {images_out}")
    print(f"📄 Annotations: {output_annotation_file}")
    print(f"📊 Summary: {summary_file}")
    print(f"\nStatistics:")
    print(f"  - Images: {len(selected_images)}")
    print(f"  - Annotations: {len(filtered_annotations)}")
    print(f"  - Average annotations per image: {len(filtered_annotations)/len(selected_images):.2f}")

if __name__ == "__main__":
    create_coco_subset()
