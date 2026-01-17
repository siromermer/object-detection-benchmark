"""
Parity check between PyTorch and ONNX model outputs
"""
from typing import List, Dict, Tuple
import numpy as np


class ParityChecker:
    """
    Compare outputs from PyTorch and ONNX models
    """
    
    def __init__(
        self,
        bbox_tolerance: float = 0.01,
        conf_tolerance: float = 0.05,
        iou_threshold: float = 0.5
    ):
        """
        Args:
            bbox_tolerance: Maximum allowed difference in bbox coordinates (relative)
            conf_tolerance: Maximum allowed difference in confidence scores
            iou_threshold: IoU threshold for matching detections
        """
        self.bbox_tolerance = bbox_tolerance
        self.conf_tolerance = conf_tolerance
        self.iou_threshold = iou_threshold
    
    def check_parity(
        self,
        pytorch_predictions: List[Dict],
        onnx_predictions: List[Dict]
    ) -> Dict:
        """
        Check parity between PyTorch and ONNX predictions
        
        Args:
            pytorch_predictions: List of PyTorch predictions
                Each dict has keys: 'bbox', 'confidence', 'class_id'
            onnx_predictions: List of ONNX predictions
                Same format as pytorch_predictions
            
        Returns:
            Dictionary with parity check results
        """
        results = {
            'num_pytorch_detections': len(pytorch_predictions),
            'num_onnx_detections': len(onnx_predictions),
            'num_matched': 0,
            'num_pytorch_only': 0,
            'num_onnx_only': 0,
            'bbox_differences': [],
            'conf_differences': [],
            'max_bbox_diff': 0.0,
            'max_conf_diff': 0.0,
            'mean_bbox_diff': 0.0,
            'mean_conf_diff': 0.0,
            'passed': False
        }
        
        if len(pytorch_predictions) == 0 and len(onnx_predictions) == 0:
            results['passed'] = True
            return results
        
        # Match detections between PyTorch and ONNX
        matches = self._match_detections(pytorch_predictions, onnx_predictions)
        
        results['num_matched'] = len(matches)
        results['num_pytorch_only'] = len(pytorch_predictions) - len(matches)
        results['num_onnx_only'] = len(onnx_predictions) - len(matches)
        
        # Calculate differences for matched detections
        for pt_idx, onnx_idx in matches:
            pt_pred = pytorch_predictions[pt_idx]
            onnx_pred = onnx_predictions[onnx_idx]
            
            # Calculate bbox difference
            bbox_diff = self._calculate_bbox_difference(
                pt_pred['bbox'],
                onnx_pred['bbox']
            )
            results['bbox_differences'].append(bbox_diff)
            
            # Calculate confidence difference
            conf_diff = abs(pt_pred['confidence'] - onnx_pred['confidence'])
            results['conf_differences'].append(conf_diff)
        
        if len(matches) > 0:
            results['max_bbox_diff'] = max(results['bbox_differences'])
            results['max_conf_diff'] = max(results['conf_differences'])
            results['mean_bbox_diff'] = np.mean(results['bbox_differences'])
            results['mean_conf_diff'] = np.mean(results['conf_differences'])
            
            # Check if differences are within tolerance
            bbox_passed = results['max_bbox_diff'] < self.bbox_tolerance
            conf_passed = results['max_conf_diff'] < self.conf_tolerance
            count_passed = abs(len(pytorch_predictions) - len(onnx_predictions)) <= 5
            
            results['passed'] = bbox_passed and conf_passed and count_passed
        else:
            results['passed'] = False
        
        return results
    
    def _match_detections(
        self,
        predictions1: List[Dict],
        predictions2: List[Dict]
    ) -> List[Tuple[int, int]]:
        """
        Match detections between two prediction sets using IoU
        
        Args:
            predictions1: First set of predictions
            predictions2: Second set of predictions
            
        Returns:
            List of (index1, index2) tuples for matched detections
        """
        matches = []
        used_indices2 = set()
        
        for i, pred1 in enumerate(predictions1):
            best_iou = 0.0
            best_j = -1
            
            for j, pred2 in enumerate(predictions2):
                if j in used_indices2:
                    continue
                
                # Only match same class
                if pred1['class_id'] != pred2['class_id']:
                    continue
                
                # Calculate IoU
                iou = self._calculate_iou(pred1['bbox'], pred2['bbox'])
                
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_j = j
            
            if best_j >= 0:
                matches.append((i, best_j))
                used_indices2.add(best_j)
        
        return matches
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_bbox_difference(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> float:
        """
        Calculate normalized difference between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            Normalized difference (0 to 1)
        """
        # Type: np.ndarray, Shape: (4,)
        bbox1_arr = np.array(bbox1)
        bbox2_arr = np.array(bbox2)
        
        # Calculate relative difference
        # Type: np.ndarray, Shape: (4,)
        diff = np.abs(bbox1_arr - bbox2_arr)
        
        # Normalize by bbox size
        w1 = bbox1[2] - bbox1[0]
        h1 = bbox1[3] - bbox1[1]
        size = max(w1, h1, 1.0)
        
        # Type: float
        normalized_diff = np.mean(diff) / size
        
        return float(normalized_diff)
    
    def print_report(self, results: Dict) -> None:
        """
        Print a formatted parity check report
        
        Args:
            results: Results from check_parity
        """
        
        print(f"\nDetection Counts:")
        print(f"  PyTorch detections: {results['num_pytorch_detections']}")
        print(f"  ONNX detections:    {results['num_onnx_detections']}")
        print(f"  Matched:            {results['num_matched']}")
        print(f"  PyTorch only:       {results['num_pytorch_only']}")
        print(f"  ONNX only:          {results['num_onnx_only']}")
        
        if results['num_matched'] > 0:
            print(f"\nBounding Box Differences:")
            print(f"  Max:  {results['max_bbox_diff']:.6f} (tolerance: {self.bbox_tolerance})")
            print(f"  Mean: {results['mean_bbox_diff']:.6f}")
            
            print(f"\nConfidence Differences:")
            print(f"  Max:  {results['max_conf_diff']:.6f} (tolerance: {self.conf_tolerance})")
            print(f"  Mean: {results['mean_conf_diff']:.6f}")
        
        print(f"\nParity Check: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")
        print("="*60 + "\n")
