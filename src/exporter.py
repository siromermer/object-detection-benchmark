"""
Model export utilities for converting PyTorch models to ONNX
"""
from pathlib import Path
import torch
from ultralytics import YOLO


class ModelExporter:
    """
    Export PyTorch models to different formats
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to PyTorch model (.pt file)
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Type: YOLO
        self.model = YOLO(str(self.model_path))
    
    def export_to_onnx(
        self,
        output_path: str,
        imgsz: int = 640,
        simplify: bool = True,
        opset: int = 12,
        int8: bool = False
    ) -> Path:
        """
        Export model to ONNX format
        
        Args:
            output_path: Output path for ONNX model
            imgsz: Input image size
            simplify: Whether to simplify the ONNX model
            opset: ONNX opset version
            int8: Whether to use INT8 quantization
            
        Returns:
            Path to exported ONNX model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        precision = "INT8" if int8 else "FP32"
        print(f"Exporting {self.model_path} to ONNX ({precision})...")
        print(f"  Input size: {imgsz}x{imgsz}")
        print(f"  Opset: {opset}")
        print(f"  Simplify: {simplify}")
        print(f"  Quantization: {precision}")
        
        # Export using ultralytics built-in export
        exported_model = self.model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=simplify,
            opset=opset,
            int8=int8
        )
        
        # Move to desired location if different
        exported_path = Path(exported_model)
        if exported_path != output_path:
            if output_path.exists():
                output_path.unlink()
            exported_path.rename(output_path)
        
        print(f"Model exported to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return output_path
    
    def export_to_torchscript(
        self,
        output_path: str,
        imgsz: int = 640,
        optimize: bool = True
    ) -> Path:
        """
        Export model to TorchScript format
        
        Args:
            output_path: Output path for TorchScript model
            imgsz: Input image size
            optimize: Whether to optimize the model
            
        Returns:
            Path to exported model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Exporting {self.model_path} to TorchScript...")
        print(f"  Input size: {imgsz}x{imgsz}")
        print(f"  Optimize: {optimize}")
        
        exported_model = self.model.export(
            format='torchscript',
            imgsz=imgsz,
            optimize=optimize
        )
        
        exported_path = Path(exported_model)
        if exported_path != output_path:
            if output_path.exists():
                output_path.unlink()
            exported_path.rename(output_path)
        
        print(f"Model exported to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return output_path
    
    def verify_onnx_export(self, onnx_path: str) -> bool:
        """
        Verify ONNX model can be loaded and has correct structure
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if verification successful
        """
        import onnx
        import onnxruntime as ort
        
        onnx_path = Path(onnx_path)
        
        if not onnx_path.exists():
            print(f" ONNX file not found: {onnx_path}")
            return False
        
        try:
            # Load and check ONNX model
            # Type: onnx.ModelProto
            onnx_model = onnx.load(str(onnx_path))
            
            # Check model validity
            onnx.checker.check_model(onnx_model)
            print("ONNX model structure is valid")
            
            # Check with ONNX Runtime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Type: onnxruntime.InferenceSession
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # Print model info
            print(f" ONNX Runtime can load the model")
            print(f"  Provider: {session.get_providers()[0]}")
            
            # Type: list[onnxruntime.NodeArg]
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            print(f"  Input: {inputs[0].name}, shape: {inputs[0].shape}, type: {inputs[0].type}")
            print(f"  Output: {outputs[0].name}, shape: {outputs[0].shape}, type: {outputs[0].type}")
            
            return True
            
        except Exception as e:
            print(f" ONNX verification failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the PyTorch model
        
        Returns:
            Dictionary with model information
        """
        # Type: torch.nn.Module
        model_module = self.model.model
        
        # Count parameters
        # Type: int
        total_params = sum(p.numel() for p in model_module.parameters())
        trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
        
        # Get model size
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        info = {
            'model_name': self.model_path.name,
            'model_path': str(self.model_path),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'device': next(model_module.parameters()).device.type
        }
        
        return info
