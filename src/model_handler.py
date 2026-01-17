"""
Model handler for PyTorch and ONNX inference
"""
import time
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO


class ModelHandler:
    """
    Handle model loading and inference for both PyTorch and ONNX backends
    """
    
    def __init__(
        self,
        model_path: str,
        backend: str = 'pytorch',
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        precision: str = 'fp32'
    ):
        """
        Args:
            model_path: Path to model file (.pt or .onnx or .torchscript)
            backend: 'pytorch' or 'onnx' or 'torchscript'
            imgsz: Input image size
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            precision: Model precision ('fp32' or 'int8')
        """
        self.model_path = Path(model_path)
        self.backend = backend
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.precision = precision
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        if backend == 'pytorch':
            self.model = self._load_pytorch_model()
        elif backend == 'torchscript':
            self.model = self._load_torchscript_model()
        elif backend == 'onnx':
            self.session = self._load_onnx_model()
            self.input_name = self.session.get_inputs()[0].name
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _load_pytorch_model(self) -> YOLO:
        """Load YOLOv8 PyTorch model"""
        model = YOLO(str(self.model_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(f"Loaded PyTorch model from {self.model_path}")
        print(f"Device: {device}")
        return model
    
    def _load_onnx_model(self) -> ort.InferenceSession:
        """Load ONNX model with GPU support if available"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(self.model_path), providers=providers)
        print(f"Loaded ONNX model from {self.model_path}")
        print(f"Providers: {session.get_providers()}")
        return session
    
    def predict(
        self,
        image_path: str,
        warmup: bool = False
    ) -> Tuple[List[Dict], float]:
        """
        Run inference on single image
        
        Args:
            image_path: Path to image file
            warmup: If True, skip time measurement
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        if self.backend == 'pytorch':
            return self._predict_pytorch(image_path, warmup)
        elif self.backend == 'onnx':
            return self._predict_onnx(image_path, warmup)
    
    def _predict_pytorch(
        self,
        image_path: str,
        warmup: bool
    ) -> Tuple[List[Dict], float]:
        """PyTorch inference using Ultralytics"""
        start_time = time.time()
        
        results = self.model.predict(
            source=image_path,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        if warmup:
            inference_time = 0.0
        
        predictions = []
        result = results[0]
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            predictions.append({
                'bbox': box.tolist(),
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': result.names[cls_id]
            })
        
        return predictions, inference_time
    
    def _predict_onnx(
        self,
        image_path: str,
        warmup: bool
    ) -> Tuple[List[Dict], float]:
        """
        ONNX inference using direct ONNX Runtime
        Uses letterbox preprocessing to match PyTorch behavior
        """
        # Load image
        img_bgr = cv2.imread(image_path)
        original_shape = img_bgr.shape[:2]  # (height, width)
        
        # Letterbox preprocessing (same as PyTorch/Ultralytics)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_letterbox, ratio, (pad_w, pad_h) = self._letterbox(img_rgb, self.imgsz)
        
        # Normalize and transpose
        img_normalized = img_letterbox.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: img_batch})
        inference_time = time.time() - start_time
        
        if warmup:
            inference_time = 0.0
        
        # Postprocess with letterbox info for correct coordinate scaling
        predictions = self._postprocess_onnx(outputs, original_shape, ratio, (pad_w, pad_h))
        
        return predictions, inference_time
    
    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize and pad image while maintaining aspect ratio (letterbox)
        Same as Ultralytics preprocessing
        
        Args:
            img: Input image, shape (H, W, 3)
            new_shape: Target size
            color: Padding color (gray by default, same as YOLO)
            
        Returns:
            img: Letterboxed image
            ratio: Scale ratio
            (pad_w, pad_h): Padding amounts
        """
        shape = img.shape[:2]  # (height, width)
        
        # Scale ratio (new / old)
        r = min(new_shape / shape[0], new_shape / shape[1])
        
        # Compute new unpadded dimensions
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # Compute padding
        dw = new_shape - new_unpad[0]  # width padding
        dh = new_shape - new_unpad[1]  # height padding
        
        # Divide padding into 2 sides
        dw /= 2
        dh /= 2
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add border (padding)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, (dw, dh)
    
    def _postprocess_onnx(
        self,
        outputs: List[np.ndarray],
        original_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float]
    ) -> List[Dict]:
        """
        Post-process YOLOv8 ONNX output with letterbox coordinate correction
        
        Args:
            outputs: Raw ONNX outputs, shape [1, 84, 8400]
            original_shape: (height, width) of original image
            ratio: Scale ratio used in letterbox
            pad: (pad_w, pad_h) padding amounts
        """
        predictions_raw = outputs[0]  # Shape: [1, 84, 8400]
        predictions_raw = predictions_raw[0]  # Remove batch: [84, 8400]
        predictions_raw = predictions_raw.T  # Transpose: [8400, 84]
        
        # Extract boxes (first 4 columns) and scores (remaining 80 columns)
        boxes = predictions_raw[:, :4]
        scores = predictions_raw[:, 4:]
        
        # Get class with highest score for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        valid_mask = confidences > self.conf_threshold
        boxes = boxes[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.column_stack((x1, y1, x2, y2))
        
        # Scale boxes back to original image coordinates (reverse letterbox)
        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w  # Remove x padding
        boxes[:, [1, 3]] -= pad_h  # Remove y padding
        boxes[:, :4] /= ratio      # Scale back to original size
        
        # Clip to image boundaries
        orig_h, orig_w = original_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        
        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            confidences.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        indices = indices.flatten()
        boxes = boxes[indices]
        confidences = confidences[indices]
        class_ids = class_ids[indices]
        
        # Build predictions list
        predictions = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            predictions.append({
                'bbox': box.tolist(),
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
            })
        
        return predictions
