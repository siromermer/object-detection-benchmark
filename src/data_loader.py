"""
Data loader for COCO dataset
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image


class COCODataLoader:
    """
    Load COCO format dataset
    """
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        num_images: int = None
    ):
        """
        Args:
            images_dir: Path to images directory
            annotations_file: Path to COCO format JSON annotations
            num_images: Limit number of images (None for all)
        """
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.num_images = num_images
        
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Type: dict[int, str], maps category_id to name
        self.categories = {
            cat['id']: cat['name'] 
            for cat in self.coco_data['categories']
        }
        
        # Type: list[dict], COCO image entries
        self.images = self.coco_data['images']
        if num_images is not None:
            self.images = self.images[:num_images]
        
        # Type: dict[int, list[dict]], maps image_id to annotations
        self.img_to_anns = self._build_annotation_index()
    
    def _build_annotation_index(self) -> Dict[int, List[Dict]]:
        """
        Build mapping from image_id to list of annotations
        
        Returns:
            Dictionary mapping image_id to annotation list
        """
        img_ids = {img['id'] for img in self.images}
        img_to_anns = {img_id: [] for img_id in img_ids}
        
        for ann in self.coco_data['annotations']:
            if ann['image_id'] in img_ids:
                img_to_anns[ann['image_id']].append(ann)
        
        return img_to_anns
    
    def __len__(self) -> int:
        """Return number of images"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Get image and metadata by index
        
        Args:
            idx: Image index
            
        Returns:
            Tuple of (image_array, metadata)
            image_array: Type np.ndarray, shape (H, W, 3)
            metadata: Type dict with keys 'image_info', 'annotations'
        """
        img_info = self.images[idx]
        img_path = self.images_dir / img_info['file_name']
        
        # Type: PIL.Image.Image
        img = Image.open(img_path).convert('RGB')
        
        # Type: np.ndarray, Shape: (H, W, 3)
        img_array = np.array(img)
        
        # Type: list[dict]
        annotations = self.img_to_anns[img_info['id']]
        
        metadata = {
            'image_info': img_info,
            'annotations': annotations,
            'categories': self.categories
        }
        
        return img_array, metadata
    
    def get_image_path(self, idx: int) -> Path:
        """
        Get path to image file
        
        Args:
            idx: Image index
            
        Returns:
            Path object
        """
        img_info = self.images[idx]
        return self.images_dir / img_info['file_name']
    
    def get_annotations_for_evaluation(self) -> Dict:
        """
        Get annotations in format suitable for pycocotools evaluation
        
        Returns:
            Dictionary with 'images', 'annotations', 'categories'
        """
        img_ids = {img['id'] for img in self.images}
        
        filtered_anns = [
            ann for ann in self.coco_data['annotations']
            if ann['image_id'] in img_ids
        ]
        
        return {
            'images': self.images,
            'annotations': filtered_anns,
            'categories': self.coco_data['categories'],
            'info': self.coco_data.get('info', {}),
            'licenses': self.coco_data.get('licenses', [])
        }
