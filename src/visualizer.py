"""
Visualization utilities for generating plots and charts
"""
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Visualizer:
    """
    Generate plots and visualizations for benchmark results
    """
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_latency_comparison(
        self,
        results_list: List[Dict],
        save_path: str = None
    ) -> Path:
        """
        Create bar chart comparing latencies across configurations
        
        Args:
            results_list: List of benchmark results
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.output_dir / "latency_comparison.png"
        else:
            save_path = Path(save_path)
        
        # Prepare data
        configs = [f"{r['backend']}_{r['imgsz']}" for r in results_list]
        means = [r['mean_latency_ms'] for r in results_list]
        stds = [r['std_latency_ms'] for r in results_list]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved latency comparison plot to {save_path}")
        return save_path
    
    def plot_throughput_comparison(
        self,
        results_list: List[Dict],
        save_path: str = None
    ) -> Path:
        """
        Create bar chart comparing throughput across configurations
        
        Args:
            results_list: List of benchmark results
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.output_dir / "throughput_comparison.png"
        else:
            save_path = Path(save_path)
        
        # Prepare data
        configs = [f"{r['backend']}_{r['imgsz']}" for r in results_list]
        throughputs = [r['throughput_fps'] for r in results_list]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, throughputs, alpha=0.7, color='forestgreen')
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (FPS)', fontsize=12, fontweight='bold')
        ax.set_title('Inference Throughput Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved throughput comparison plot to {save_path}")
        return save_path
    
    def plot_accuracy_latency_tradeoff(
        self,
        results_list: List[Dict],
        save_path: str = None
    ) -> Path:
        """
        Create scatter plot showing accuracy vs latency tradeoff
        
        Args:
            results_list: List of results with 'metrics' and benchmark data
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.output_dir / "accuracy_latency_tradeoff.png"
        else:
            save_path = Path(save_path)
        
        # Prepare data
        configs = []
        latencies = []
        accuracies = []
        
        for r in results_list:
            if 'metrics' in r:
                configs.append(f"{r['backend']}_{r['imgsz']}")
                latencies.append(r['mean_latency_ms'])
                accuracies.append(r['metrics'].get('mAP_50', 0) * 100)
        
        if len(configs) == 0:
            print("Warning: No metrics found for accuracy-latency plot")
            return save_path
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            latencies, accuracies,
            s=200, alpha=0.6, c=range(len(configs)),
            cmap='viridis', edgecolors='black', linewidth=1.5
        )
        
        # Add labels for each point
        for i, config in enumerate(configs):
            ax.annotate(
                config,
                (latencies[i], accuracies[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5)
            )
        
        ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5 (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy vs Latency Tradeoff', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved accuracy-latency tradeoff plot to {save_path}")
        return save_path
    
    def plot_latency_distribution(
        self,
        results: Dict,
        save_path: str = None
    ) -> Path:
        """
        Create histogram showing latency distribution
        
        Args:
            results: Benchmark results
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            config_name = f"{results['backend']}_{results['imgsz']}"
            save_path = self.output_dir / f"latency_distribution_{config_name}.png"
        else:
            save_path = Path(save_path)
        
        # Type: list[float]
        latencies = results['latencies_ms']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(latencies, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add mean line
        mean_latency = results['mean_latency_ms']
        ax.axvline(
            mean_latency, color='red', linestyle='--',
            linewidth=2, label=f'Mean: {mean_latency:.2f} ms'
        )
        
        # Add median line
        median_latency = results['median_latency_ms']
        ax.axvline(
            median_latency, color='green', linestyle='--',
            linewidth=2, label=f'Median: {median_latency:.2f} ms'
        )
        
        ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(
            f"Latency Distribution - {results['backend']}_{results['imgsz']}",
            fontsize=14, fontweight='bold'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved latency distribution plot to {save_path}")
        return save_path
    
    def plot_metrics_comparison(
        self,
        results_list: List[Dict],
        save_path: str = None
    ) -> Path:
        """
        Create grouped bar chart comparing multiple metrics
        
        Args:
            results_list: List of results with 'metrics'
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = self.output_dir / "metrics_comparison.png"
        else:
            save_path = Path(save_path)
        
        # Prepare data
        configs = [f"{r['backend']}_{r['imgsz']}" for r in results_list]
        
        metrics_to_plot = ['mAP', 'mAP_50', 'mAP_75']
        data = {metric: [] for metric in metrics_to_plot}
        
        for r in results_list:
            if 'metrics' in r:
                for metric in metrics_to_plot:
                    data[metric].append(r['metrics'].get(metric, 0) * 100)
            else:
                for metric in metrics_to_plot:
                    data[metric].append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(configs))
        width = 0.25
        
        colors = ['steelblue', 'forestgreen', 'coral']
        
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            offset = width * (i - 1)
            ax.bar(x + offset, data[metric], width, label=metric, alpha=0.8, color=color)
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Detection Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved metrics comparison plot to {save_path}")
        return save_path
    
    def plot_parity_check_results(
        self,
        parity_results: Dict,
        save_dir: Path = None
    ) -> List[Path]:
        """
        Create parity check specific visualizations
        
        Args:
            parity_results: Aggregate parity check results
            save_dir: Directory to save plots
            
        Returns:
            List of paths to saved plots
        """
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
        
        saved_plots = []
        
        # 1. Bbox differences histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_bbox_diffs = []
        for result in parity_results['per_image_results']:
            if result['bbox_differences']:
                all_bbox_diffs.extend(result['bbox_differences'])
        
        if all_bbox_diffs:
            ax.hist(all_bbox_diffs, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.axvline(parity_results['tolerance_thresholds']['bbox_tolerance'], 
                      color='red', linestyle='--', linewidth=2, 
                      label=f"Tolerance: {parity_results['tolerance_thresholds']['bbox_tolerance']}")
            ax.axvline(parity_results['bbox_stats']['mean'], 
                      color='green', linestyle='--', linewidth=2, 
                      label=f"Mean: {parity_results['bbox_stats']['mean']:.6f}")
            
            ax.set_xlabel('Relative Bounding Box Difference', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Parity Check: Bounding Box Differences Distribution\n(PyTorch vs ONNX)', 
                        fontweight='bold', fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            bbox_plot = save_dir / "parity_bbox_differences.png"
            plt.savefig(bbox_plot, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(bbox_plot)
            print(f"✓ Saved bbox differences plot to {bbox_plot}")
        
        # 2. Confidence differences histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_conf_diffs = []
        for result in parity_results['per_image_results']:
            if result['conf_differences']:
                all_conf_diffs.extend(result['conf_differences'])
        
        if all_conf_diffs:
            ax.hist(all_conf_diffs, bins=50, color='#FFA500', alpha=0.7, edgecolor='black')
            ax.axvline(parity_results['tolerance_thresholds']['conf_tolerance'], 
                      color='red', linestyle='--', linewidth=2, 
                      label=f"Tolerance: {parity_results['tolerance_thresholds']['conf_tolerance']}")
            ax.axvline(parity_results['conf_stats']['mean'], 
                      color='green', linestyle='--', linewidth=2, 
                      label=f"Mean: {parity_results['conf_stats']['mean']:.6f}")
            
            ax.set_xlabel('Absolute Confidence Difference', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Parity Check: Confidence Differences Distribution\n(PyTorch vs ONNX)', 
                        fontweight='bold', fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            conf_plot = save_dir / "parity_conf_differences.png"
            plt.savefig(conf_plot, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(conf_plot)
            print(f"✓ Saved confidence differences plot to {conf_plot}")
        
        # 3. Detection count comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Total\nDetections', 'Matched\nDetections', 'PyTorch\nOnly', 'ONNX\nOnly']
        pytorch_only = sum(r['num_pytorch_only'] for r in parity_results['per_image_results'])
        onnx_only = sum(r['num_onnx_only'] for r in parity_results['per_image_results'])
        
        values = [
            parity_results['total_pytorch_detections'],
            parity_results['total_matched'],
            pytorch_only,
            onnx_only
        ]
        colors = ['#2E86AB', '#28A745', '#FFC107', '#DC3545']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Count', fontweight='bold', fontsize=12)
        ax.set_title('Parity Check: Detection Counts Comparison\n(PyTorch vs ONNX)', 
                    fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add match rate annotation
        ax.text(0.98, 0.95, f"Match Rate: {parity_results['match_rate']*100:.1f}%", 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        detcount_plot = save_dir / "parity_detection_counts.png"
        plt.savefig(detcount_plot, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(detcount_plot)
        print(f"✓ Saved detection counts plot to {detcount_plot}")
        
        return saved_plots

