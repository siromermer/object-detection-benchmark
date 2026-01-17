"""
Main benchmark CLI for object detection model evaluation
Usage:
    python benchmark.py --backend pytorch --images 300 --out results/
    python benchmark.py --backend onnx --images 300 --out results/
"""
import argparse
import json
import yaml
from pathlib import Path
from typing import List
import pandas as pd
from src.data_loader import COCODataLoader
from src.model_handler import ModelHandler
from src.evaluator import Evaluator
from src.exporter import ModelExporter
from src.parity_check import ParityChecker
from src.benchmarker import Benchmarker
from src.visualizer import Visualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Benchmark object detection models (PyTorch/ONNX)"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["pytorch", "onnx", "all"],
        help="Backend to use for inference"
    )
    
    parser.add_argument(
        "--images",
        type=int,
        default=500,
        help="Number of images to process"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export PyTorch model to ONNX before benchmarking"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_model(model_name: str, model_path: str) -> Path:
    """Download pretrained model if not exists"""
    from ultralytics import YOLO
    
    model_path = Path(model_path)
    
    if model_path.exists():
        print(f" Model already exists: {model_path}")
        return model_path
    
    print(f"Downloading {model_name} model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download model
    model = YOLO(f"{model_name}.pt")
    model.save(str(model_path))
    
    print(f" Model downloaded: {model_path}")
    return model_path


def run_benchmark(
    backend: str,
    config: dict,
    num_images: int,
    output_dir: Path,
    imgsz: int
):
    """
    Run benchmark for specified backend
    
    Args:
        backend: 'pytorch' or 'onnx'
        config: Configuration dictionary
        num_images: Number of images to process
        output_dir: Output directory
        imgsz: Input image size
    """
    print(f"BENCHMARKING: {backend.upper()} ({imgsz}x{imgsz})")
    
    # Load dataset
    print("\nLoading dataset...")
    data_loader = COCODataLoader(
        images_dir=config['dataset']['images_dir'],
        annotations_file=config['dataset']['annotations_file'],
        num_images=num_images
    )
    print(f" Loaded {len(data_loader)} images")
    
    # Get image paths
    image_paths = [str(data_loader.get_image_path(i)) for i in range(len(data_loader))]
    
    # Initialize model
    if backend == 'pytorch':
        model_path = config['model']['path']
    elif backend == 'onnx':
        onnx_base = Path(config['model']['onnx_path'])
        model_path = str(onnx_base.parent / f"{onnx_base.stem}_{imgsz}{onnx_base.suffix}")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    if not Path(model_path).exists():
        print(f" Model not found: {model_path}")
        return None
    
    print(f"\nLoading {backend} model...")
    model_handler = ModelHandler(
        model_path=model_path,
        backend=backend,
        imgsz=imgsz,
        conf_threshold=config['inference']['conf_threshold'],
        iou_threshold=config['inference']['iou_threshold']
    )
    
    # Run benchmark
    benchmarker = Benchmarker(
        warmup_runs=config['benchmark']['warmup_runs'],
        measurement_runs=config['benchmark']['measurement_runs']
    )
    
    results = benchmarker.benchmark_model(
        model_handler=model_handler,
        image_paths=image_paths,
        verbose=True
    )
    
    # Run evaluation
    print("\nEvaluating detection accuracy...")
    evaluator = Evaluator(data_loader.get_annotations_for_evaluation())
    
    # Convert predictions to COCO format
    predictions_dict = {}
    for img_path, preds in results['predictions'].items():
        img_idx = image_paths.index(img_path)
        img_id = data_loader.images[img_idx]['id']
        predictions_dict[img_id] = preds
    
    coco_predictions = evaluator.format_predictions_for_coco(predictions_dict)
    
    # Calculate metrics
    metrics = evaluator.evaluate(coco_predictions)
    results['metrics'] = metrics
    
    print(f"\n✓ mAP@0.5: {metrics['mAP_50']:.4f}")
    print(f"✓ mAP@0.5:0.95: {metrics['mAP']:.4f}")
    
    # Save results
    results_file = output_dir / f"results_{backend}_{imgsz}.json"
    
    # Save predictions separately for parity check (as pickle for efficiency)
    predictions_file = output_dir / f"predictions_{backend}_{imgsz}.pkl"
    import pickle
    with open(predictions_file, 'wb') as f:
        pickle.dump(results['predictions'], f)
    
    with open(results_file, 'w') as f:
        # Remove predictions from JSON (too large, saved as pickle)
        results_to_save = {k: v for k, v in results.items() if k != 'predictions'}
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Predictions saved to {predictions_file}")
    
    return results


def save_csv_reports(results_list: List[dict], output_dir: Path, append: bool = True):
    """
    Save metrics and latency as CSV files
    
    Args:
        results_list: List of benchmark results
        output_dir: Output directory
        append: If True, append to existing CSV files (avoiding duplicates)
    """
    metrics_file = output_dir / "metrics.csv"
    latency_file = output_dir / "latency.csv"
    
    # Metrics CSV
    metrics_data = []
    for r in results_list:
        if 'metrics' in r:
            row = {
                'backend': r['backend'],
                'imgsz': r['imgsz'],
                'mAP': r['metrics']['mAP'],
                'mAP_50': r['metrics']['mAP_50'],
                'mAP_75': r['metrics']['mAP_75'],
                'mAP_small': r['metrics']['mAP_small'],
                'mAP_medium': r['metrics']['mAP_medium'],
                'mAP_large': r['metrics']['mAP_large']
            }
            metrics_data.append(row)
    
    if metrics_data:
        df_new = pd.DataFrame(metrics_data)
        
        if append and metrics_file.exists():
            df_existing = pd.read_csv(metrics_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['backend', 'imgsz'], keep='last')
        else:
            df_combined = df_new
        
        df_combined.to_csv(metrics_file, index=False)
        print(f"Saved metrics to {metrics_file}")
    
    # Latency CSV
    latency_data = []
    for r in results_list:
        row = {
            'backend': r['backend'],
            'imgsz': r['imgsz'],
            'mean_latency_ms': r['mean_latency_ms'],
            'std_latency_ms': r['std_latency_ms'],
            'min_latency_ms': r['min_latency_ms'],
            'max_latency_ms': r['max_latency_ms'],
            'median_latency_ms': r['median_latency_ms'],
            'p95_latency_ms': r['p95_latency_ms'],
            'p99_latency_ms': r['p99_latency_ms'],
            'throughput_fps': r['throughput_fps']
        }
        latency_data.append(row)
    
    df_new = pd.DataFrame(latency_data)
    
    if append and latency_file.exists():
        df_existing = pd.read_csv(latency_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['backend', 'imgsz'], keep='last')
    else:
        df_combined = df_new
    
    df_combined.to_csv(latency_file, index=False)
    print(f"Saved latency data to {latency_file}")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Backend: {args.backend}")
    print(f"Images: {args.images}")
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"Output: {output_dir}")
    
    # Download model if needed
    model_path = download_model(config['model']['name'], config['model']['path'])
    
    # Export to ONNX if requested (with size-specific naming)
    if args.export or args.backend in ['onnx', 'all']:
        onnx_base = Path(config['model']['onnx_path'])
        onnx_path = onnx_base.parent / f"{onnx_base.stem}_{args.imgsz}{onnx_base.suffix}"
        
        if not onnx_path.exists() or args.export:
            print(f"\nExporting model to ONNX ({args.imgsz}x{args.imgsz})...")
            exporter = ModelExporter(str(model_path))
            exporter.export_to_onnx(
                output_path=str(onnx_path),
                imgsz=args.imgsz
            )
            exporter.verify_onnx_export(str(onnx_path))
    
    # Run benchmarks
    results_list = []
    
    if args.backend == 'all':
        # Run both backends
        for backend in ['pytorch', 'onnx']:
            result = run_benchmark(backend, config, args.images, output_dir, args.imgsz)
            if result:
                results_list.append(result)
    else:
        # Run single backend
        result = run_benchmark(args.backend, config, args.images, output_dir, args.imgsz)
        if result:
            results_list.append(result)
    
    if len(results_list) == 0:
        print("\nNo results generated")
        return
    
    # Parity check - automatic when backend=all, or when --parity-check flag is used
    run_parity = (args.backend == 'all') or (args.parity_check and len(results_list) >= 2)
    
    if run_parity and len(results_list) >= 2:
        print("\nRUNNING PARITY CHECK")
        
        import pickle
        import numpy as np
        
        # Try to load saved predictions first
        pytorch_pred_file = output_dir / f"predictions_pytorch_{args.imgsz}.pkl"
        onnx_pred_file = output_dir / f"predictions_onnx_{args.imgsz}.pkl"
        
        pytorch_preds = None
        onnx_preds = None
        
        # Check if saved predictions exist
        if pytorch_pred_file.exists() and onnx_pred_file.exists():
            print(f"Using saved predictions from previous benchmark runs")
            print(f"  PyTorch: {pytorch_pred_file}")
            print(f"  ONNX: {onnx_pred_file}")
            
            with open(pytorch_pred_file, 'rb') as f:
                pytorch_preds = pickle.load(f)
            with open(onnx_pred_file, 'rb') as f:
                onnx_preds = pickle.load(f)
        else:
            # No saved predictions, need to run inference
            print(f"No saved predictions found, running inference...")
            
            data_loader = COCODataLoader(
                images_dir=config['dataset']['images_dir'],
                annotations_file=config['dataset']['annotations_file'],
                num_images=args.images
            )
            
            pytorch_handler = ModelHandler(
                model_path=config['model']['path'],
                backend='pytorch',
                imgsz=args.imgsz
            )
            
            onnx_base = Path(config['model']['onnx_path'])
            onnx_path_sized = onnx_base.parent / f"{onnx_base.stem}_{args.imgsz}{onnx_base.suffix}"
            
            onnx_handler = ModelHandler(
                model_path=str(onnx_path_sized),
                backend='onnx',
                imgsz=args.imgsz
            )
            
            pytorch_preds = {}
            onnx_preds = {}
            
            print("Running PyTorch inference...")
            for i in range(len(data_loader)):
                img_path = str(data_loader.get_image_path(i))
                preds, _ = pytorch_handler.predict(img_path)
                pytorch_preds[img_path] = preds
            
            print("Running ONNX inference...")
            for i in range(len(data_loader)):
                img_path = str(data_loader.get_image_path(i))
                preds, _ = onnx_handler.predict(img_path)
                onnx_preds[img_path] = preds
        
        # Get common images
        common_images = set(pytorch_preds.keys()) & set(onnx_preds.keys())
        num_images = len(common_images)
        print(f"Testing parity on {num_images} images...")
        
        # Check parity
        checker = ParityChecker(
            bbox_tolerance=config['parity']['bbox_tolerance'],
            conf_tolerance=config['parity']['conf_tolerance'],
            iou_threshold=config['parity']['iou_threshold']
        )
        
        all_parity_results = []
        all_bbox_diffs = []
        all_conf_diffs = []
        
        for img_path in sorted(common_images):
            parity_result = checker.check_parity(pytorch_preds[img_path], onnx_preds[img_path])
            all_parity_results.append(parity_result)
            
            if parity_result['bbox_differences']:
                all_bbox_diffs.extend(parity_result['bbox_differences'])
            if parity_result['conf_differences']:
                all_conf_diffs.extend(parity_result['conf_differences'])
        
        # Aggregate results
        import numpy as np
        total_pytorch_dets = sum(r['num_pytorch_detections'] for r in all_parity_results)
        total_onnx_dets = sum(r['num_onnx_detections'] for r in all_parity_results)
        total_matched = sum(r['num_matched'] for r in all_parity_results)
        images_passed = sum(1 for r in all_parity_results if r['passed'])
        
        aggregate_results = {
            'num_images_tested': num_images,
            'images_passed': images_passed,
            'images_failed': num_images - images_passed,
            'pass_rate': images_passed / num_images,
            'total_pytorch_detections': total_pytorch_dets,
            'total_onnx_detections': total_onnx_dets,
            'total_matched': total_matched,
            'match_rate': total_matched / max(total_pytorch_dets, 1),
            'bbox_stats': {
                'mean': float(np.mean(all_bbox_diffs)) if all_bbox_diffs else 0.0,
                'std': float(np.std(all_bbox_diffs)) if all_bbox_diffs else 0.0,
                'max': float(np.max(all_bbox_diffs)) if all_bbox_diffs else 0.0,
                'min': float(np.min(all_bbox_diffs)) if all_bbox_diffs else 0.0,
                'p95': float(np.percentile(all_bbox_diffs, 95)) if all_bbox_diffs else 0.0,
                'p99': float(np.percentile(all_bbox_diffs, 99)) if all_bbox_diffs else 0.0
            },
            'conf_stats': {
                'mean': float(np.mean(all_conf_diffs)) if all_conf_diffs else 0.0,
                'std': float(np.std(all_conf_diffs)) if all_conf_diffs else 0.0,
                'max': float(np.max(all_conf_diffs)) if all_conf_diffs else 0.0,
                'min': float(np.min(all_conf_diffs)) if all_conf_diffs else 0.0,
                'p95': float(np.percentile(all_conf_diffs, 95)) if all_conf_diffs else 0.0,
                'p99': float(np.percentile(all_conf_diffs, 99)) if all_conf_diffs else 0.0
            },
            'tolerance_thresholds': {
                'bbox_tolerance': config['parity']['bbox_tolerance'],
                'conf_tolerance': config['parity']['conf_tolerance'],
                'iou_threshold': config['parity']['iou_threshold']
            },
            'per_image_results': all_parity_results
        }
        
        # Print aggregate report
        print("\nPARITY CHECK AGGREGATE RESULTS")
        print(f"\nImages Tested: {aggregate_results['num_images_tested']}")
        print(f"Images Passed: {aggregate_results['images_passed']} ({aggregate_results['pass_rate']*100:.1f}%)")
        print(f"Images Failed: {aggregate_results['images_failed']} ({(1-aggregate_results['pass_rate'])*100:.1f}%)")
        
        print(f"\nDetection Counts:")
        print(f"  PyTorch: {aggregate_results['total_pytorch_detections']}")
        print(f"  ONNX:    {aggregate_results['total_onnx_detections']}")
        print(f"  Matched: {aggregate_results['total_matched']} ({aggregate_results['match_rate']*100:.1f}%)")
        
        print(f"\nBounding Box Differences (relative):")
        print(f"  Mean: {aggregate_results['bbox_stats']['mean']:.6f}")
        print(f"  Std:  {aggregate_results['bbox_stats']['std']:.6f}")
        print(f"  Max:  {aggregate_results['bbox_stats']['max']:.6f} (tolerance: {config['parity']['bbox_tolerance']})")
        print(f"  P95:  {aggregate_results['bbox_stats']['p95']:.6f}")
        print(f"  P99:  {aggregate_results['bbox_stats']['p99']:.6f}")
        
        print(f"\nConfidence Differences (absolute):")
        print(f"  Mean: {aggregate_results['conf_stats']['mean']:.6f}")
        print(f"  Std:  {aggregate_results['conf_stats']['std']:.6f}")
        print(f"  Max:  {aggregate_results['conf_stats']['max']:.6f} (tolerance: {config['parity']['conf_tolerance']})")
        print(f"  P95:  {aggregate_results['conf_stats']['p95']:.6f}")
        print(f"  P99:  {aggregate_results['conf_stats']['p99']:.6f}")
        
        overall_pass = aggregate_results['pass_rate'] >= 0.95
        print(f"\nOverall Parity Check: {'PASSED' if overall_pass else 'FAILED'}\n")
        
        # Save parity results
        parity_file = output_dir / f"parity_check_{args.imgsz}.json"
        with open(parity_file, 'w') as f:
            json.dump(aggregate_results, f, indent=2)
        print(f"Parity check results saved to {parity_file}")
        
        # Generate parity-specific visualizations
        print("\nGenerating parity check visualizations...")
        parity_visualizer = Visualizer(output_dir=output_dir / "plots")
        parity_visualizer.plot_parity_check_results(aggregate_results, save_dir=output_dir / "plots")
    
    # Save CSV reports and benchmark plots (always)
    if len(results_list) > 0:
        save_csv_reports(results_list, output_dir)
        
        visualizer = Visualizer(output_dir=output_dir / "plots")
        
        if len(results_list) > 1:
            visualizer.plot_latency_comparison(results_list)
            visualizer.plot_throughput_comparison(results_list)
            visualizer.plot_accuracy_latency_tradeoff(results_list)
            visualizer.plot_metrics_comparison(results_list)
        
        for result in results_list:
            visualizer.plot_latency_distribution(result)
        
        print(f"\nAll results saved to: {output_dir}")
        print(f"Plots saved to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
