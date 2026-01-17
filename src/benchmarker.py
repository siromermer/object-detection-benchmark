"""
Benchmarking utilities for measuring inference performance
"""
import time
from typing import List, Dict
import numpy as np
from tqdm import tqdm


class Benchmarker:
    """
    Benchmark model inference performance
    """
    
    def __init__(
        self,
        warmup_runs: int = 3,
        measurement_runs: int = 10
    ):
        """
        Args:
            warmup_runs: Number of warmup iterations before measurement
            measurement_runs: Number of runs for latency measurement
        """
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
    
    def benchmark_model(
        self,
        model_handler,
        image_paths: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Benchmark model on a set of images
        
        Args:
            model_handler: ModelHandler instance
            image_paths: List of paths to images
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with benchmark results
        """
        if verbose:
            print(f"\nBenchmarking {model_handler.backend} backend...")
            print(f"  Warmup runs: {self.warmup_runs}")
            print(f"  Measurement runs: {self.measurement_runs}")
            print(f"  Number of images: {len(image_paths)}")
        
        # Warmup phase
        if verbose:
            print("\nWarmup phase...")
        
        for i in range(self.warmup_runs):
            img_path = image_paths[i % len(image_paths)]
            _ = model_handler.predict(img_path, warmup=True)
        
        # Measurement phase
        if verbose:
            print("Measurement phase...")
        
        # Type: list[float]
        latencies = []
        
        # Type: dict[str, list]
        all_predictions = {}
        
        iterator = tqdm(image_paths, desc="Processing images") if verbose else image_paths
        
        for img_path in iterator:
            # Measure latency over multiple runs
            # Type: list[float]
            run_times = []
            predictions = None
            
            for _ in range(self.measurement_runs):
                preds, inference_time = model_handler.predict(img_path, warmup=False)
                run_times.append(inference_time)
                predictions = preds
            
            # Type: float
            avg_latency = np.mean(run_times)
            latencies.append(avg_latency)
            
            # Store predictions from last run
            all_predictions[img_path] = predictions
        
        # Calculate statistics
        results = self._calculate_statistics(latencies)
        results['predictions'] = all_predictions
        results['backend'] = model_handler.backend
        results['imgsz'] = model_handler.imgsz
        results['num_images'] = len(image_paths)
        
        if verbose:
            self.print_benchmark_report(results)
        
        return results
    
    def _calculate_statistics(self, latencies: List[float]) -> Dict:
        """
        Calculate statistics from latency measurements
        
        Args:
            latencies: List of latency values in seconds
            
        Returns:
            Dictionary with statistics
        """
        # Type: np.ndarray, Shape: (N,)
        latencies_arr = np.array(latencies) * 1000  # Convert to milliseconds
        
        results = {
            'latencies_ms': latencies_arr.tolist(),
            'mean_latency_ms': float(np.mean(latencies_arr)),
            'std_latency_ms': float(np.std(latencies_arr)),
            'min_latency_ms': float(np.min(latencies_arr)),
            'max_latency_ms': float(np.max(latencies_arr)),
            'median_latency_ms': float(np.median(latencies_arr)),
            'p95_latency_ms': float(np.percentile(latencies_arr, 95)),
            'p99_latency_ms': float(np.percentile(latencies_arr, 99)),
            'throughput_fps': float(1000.0 / np.mean(latencies_arr)) if np.mean(latencies_arr) > 0 else 0.0
        }
        
        return results
    
    def print_benchmark_report(self, results: Dict) -> None:
        """
        Print formatted benchmark report
        
        Args:
            results: Results from benchmark_model
        """
        print("\n" + "="*60)
        print(f"BENCHMARK REPORT - {results['backend'].upper()} ({results['imgsz']}x{results['imgsz']})")
        print("="*60)
        
        print(f"\nLatency Statistics (ms):")
        print(f"  Mean:   {results['mean_latency_ms']:.2f} ± {results['std_latency_ms']:.2f}")
        print(f"  Median: {results['median_latency_ms']:.2f}")
        print(f"  Min:    {results['min_latency_ms']:.2f}")
        print(f"  Max:    {results['max_latency_ms']:.2f}")
        print(f"  P95:    {results['p95_latency_ms']:.2f}")
        print(f"  P99:    {results['p99_latency_ms']:.2f}")
        
        print(f"\nThroughput:")
        print(f"  {results['throughput_fps']:.2f} FPS")
        
        print(f"\nConfiguration:")
        print(f"  Images processed: {results['num_images']}")
        print(f"  Backend: {results['backend']}")
        print(f"  Input size: {results['imgsz']}x{results['imgsz']}")
        
        print("="*60 + "\n")
    
    def compare_backends(
        self,
        results_list: List[Dict]
    ) -> None:
        """
        Compare benchmark results from multiple backends
        
        Args:
            results_list: List of benchmark results
        """
        if len(results_list) < 2:
            print("Need at least 2 results to compare")
            return
        
        print("\n" + "="*60)
        print("BACKEND COMPARISON")
        print("="*60)
        
        # Create comparison table
        print(f"\n{'Configuration':<25} {'Mean (ms)':<12} {'Throughput':<12} {'Speedup':<10}")
        print("-" * 60)
        
        # Type: float
        baseline_latency = results_list[0]['mean_latency_ms']
        
        for result in results_list:
            config_name = f"{result['backend']}_{result['imgsz']}"
            mean_latency = result['mean_latency_ms']
            throughput = result['throughput_fps']
            speedup = baseline_latency / mean_latency if mean_latency > 0 else 0.0
            
            print(f"{config_name:<25} {mean_latency:>10.2f}   {throughput:>10.2f}   {speedup:>8.2f}x")
        
        print("="*60 + "\n")
