"""
Evaluation metrics for object detection
"""
import json
import tempfile
from pathlib import Path
from typing import List, Dict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

"""
COCO category ID mapping from YOLO index (0-79) to COCO category ID
YOLO uses sequential 0-79 indices, COCO uses non-sequential 1-90 IDs
"""
YOLO_TO_COCO_CLASS_ID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]


class Evaluator:
    """
    Calculate evaluation metrics using COCO evaluation tools
    """
    
    def __init__(self, ground_truth_annotations: Dict):
        """
        Args:
            ground_truth_annotations: COCO format annotations dictionary
        """
        self.gt_annotations = ground_truth_annotations
        
        # Create temporary file for ground truth
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ground_truth_annotations, f)
            self.gt_file = f.name
        
        # Type: pycocotools.coco.COCO
        self.coco_gt = COCO(self.gt_file)
    
    def evaluate(
        self,
        predictions: List[Dict],
        iou_type: str = 'bbox'
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions: List of predictions in COCO format
                Each dict has keys: 'image_id', 'category_id', 'bbox', 'score'
            iou_type: Type of IoU ('bbox' or 'segm')
            
        Returns:
            Dictionary with metric names and values
        """
        if len(predictions) == 0:
            print("Warning: No predictions to evaluate")
            return {
                'mAP': 0.0,
                'mAP_50': 0.0,
                'mAP_75': 0.0,
                'mAP_small': 0.0,
                'mAP_medium': 0.0,
                'mAP_large': 0.0
            }
        
        # Create temporary file for predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f)
            pred_file = f.name
        
        # Load predictions
        coco_dt = self.coco_gt.loadRes(pred_file)
        
        # Run evaluation
        # Type: pycocotools.cocoeval.COCOeval
        coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        # Type: np.ndarray, Shape: (12,)
        stats = coco_eval.stats
        
        metrics = {
            'mAP': float(stats[0]),           # mAP @ IoU=0.50:0.95
            'mAP_50': float(stats[1]),        # mAP @ IoU=0.50
            'mAP_75': float(stats[2]),        # mAP @ IoU=0.75
            'mAP_small': float(stats[3]),     # mAP for small objects
            'mAP_medium': float(stats[4]),    # mAP for medium objects
            'mAP_large': float(stats[5]),     # mAP for large objects
            'AR_1': float(stats[6]),          # AR given 1 detection
            'AR_10': float(stats[7]),         # AR given 10 detections
            'AR_100': float(stats[8]),        # AR given 100 detections
            'AR_small': float(stats[9]),      # AR for small objects
            'AR_medium': float(stats[10]),    # AR for medium objects
            'AR_large': float(stats[11])      # AR for large objects
        }
        
        # Clean up temporary file
        Path(pred_file).unlink(missing_ok=True)
        
        return metrics
    
    def format_predictions_for_coco(
        self,
        predictions: Dict[int, List[Dict]]
    ) -> List[Dict]:
        """
        Convert predictions to COCO evaluation format
        
        Args:
            predictions: Dict mapping image_id to list of predictions
                Each prediction has keys: 'bbox', 'confidence', 'class_id'
                bbox format: [x1, y1, x2, y2]
            
        Returns:
            List of predictions in COCO format
        """
        # Type: list[dict]
        coco_predictions = []
        
        for image_id, preds in predictions.items():
            for pred in preds:
                # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = pred['bbox']
                w = x2 - x1
                h = y2 - y1
                
                # Map YOLO class index (0-79) to COCO category ID (non-sequential 1-90)
                yolo_class_id = int(pred['class_id'])
                if 0 <= yolo_class_id < len(YOLO_TO_COCO_CLASS_ID):
                    coco_category_id = YOLO_TO_COCO_CLASS_ID[yolo_class_id]
                else:
                    coco_category_id = yolo_class_id + 1
                
                coco_predictions.append({
                    'image_id': int(image_id),
                    'category_id': coco_category_id,
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(pred['confidence'])
                })
        
        return coco_predictions
    
    def calculate_precision_recall(
        self,
        predictions: List[Dict],
        confidence_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate precision and recall at specific confidence threshold
        
        Args:
            predictions: COCO format predictions
            confidence_threshold: Confidence threshold for positive predictions
            
        Returns:
            Dictionary with precision and recall
        """
        # Filter by confidence
        filtered_preds = [
            p for p in predictions 
            if p['score'] >= confidence_threshold
        ]
        
        if len(filtered_preds) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Load predictions
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(filtered_preds, f)
            pred_file = f.name
        
        coco_dt = self.coco_gt.loadRes(pred_file)
        
        # Type: pycocotools.cocoeval.COCOeval
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.params.iouThrs = np.array([0.5])  # IoU threshold
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # Extract precision and recall
        # Type: np.ndarray, Shape: (T, R, K, A, M)
        # T=iou thresholds, R=recall thresholds, K=categories, A=areas, M=max dets
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        
        # Average over all categories and areas
        # Type: float
        mean_precision = np.mean(precision[precision > -1])
        mean_recall = np.mean(recall[recall > -1])
        
        if mean_precision + mean_recall > 0:
            f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        else:
            f1_score = 0.0
        
        # Clean up
        Path(pred_file).unlink(missing_ok=True)
        
        return {
            'precision': float(mean_precision) if not np.isnan(mean_precision) else 0.0,
            'recall': float(mean_recall) if not np.isnan(mean_recall) else 0.0,
            'f1_score': float(f1_score) if not np.isnan(f1_score) else 0.0
        }
    
    def __del__(self):
        """Clean up temporary files"""
        if hasattr(self, 'gt_file'):
            Path(self.gt_file).unlink(missing_ok=True)
