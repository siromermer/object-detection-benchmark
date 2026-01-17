"""
Master script to run all benchmark configurations
Usage:
    python run_all_benchmarks.py                    # Default: 500 images, both resolutions
    python run_all_benchmarks.py --images 300       # 300 images, both resolutions
    python run_all_benchmarks.py --imgsz 640        # Only 640x640 resolution
    python run_all_benchmarks.py --imgsz 320        # Only 320x320 resolution
"""
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd):
    """Execute command and print output"""
    print(f"\nRunning: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True

def main():
    """Run all benchmark configurations"""
    
    parser = argparse.ArgumentParser(
        description="Run benchmark suite with configurable parameters"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=500,
        help="Number of images to process (default: 500)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        choices=[320, 640],
        help="Run only specific resolution: 320 or 640 (default: both)"
    )
    args = parser.parse_args()
    
    # Determine which resolutions to run
    if args.imgsz:
        resolutions = [args.imgsz]
    else:
        resolutions = [640, 320]
    
    # Build configurations based on selected resolutions
    configurations = []
    for imgsz in resolutions:
        configurations.append(
            ["python", "benchmark.py", "--backend", "all", "--images", str(args.images), "--imgsz", str(imgsz)]
        )
    
    print("STARTING BENCHMARK SUITE")
    print(f"Resolutions: {resolutions}")
    print(f"Images per config: {args.images}")
    print(f"Configurations: {len(configurations)} (PyTorch + ONNX + Parity per resolution)")
    
    # Estimate time
    time_per_res = 6  # ~6 min per resolution (both backends + parity)
    time_estimate = len(configurations) * time_per_res
    print(f"Estimated time: ~{time_estimate} minutes")
    
    failed = []
    
    for i, cmd in enumerate(configurations, 1):
        resolution = f"{resolutions[i-1]}x{resolutions[i-1]}"
        print(f"\n\nCONFIGURATION {i}/{len(configurations)}: {resolution}")
        print(f"Running full pipeline (PyTorch + ONNX + Parity Check)...")
        if not run_command(cmd):
            failed.append(' '.join(cmd))
    
    print("\nBENCHMARK SUITE COMPLETE")
    
    if failed:
        print("\nFAILED CONFIGURATIONS:")
        for cmd in failed:
            print(f"  - {cmd}")
        return 1
    else:
        print("\nAll configurations completed successfully!")
        print("\nResults available in:")
        print("  - results/metrics.csv")
        print("  - results/latency.csv")
        print("  - results/plots/")
        print("  - results/parity_check_640.json")
        print("  - results/parity_check_320.json")
        print("\nRun analysis notebook to generate visualizations:")
        print("  jupyter notebook results_analysis.ipynb")
        return 0

if __name__ == "__main__":
    sys.exit(main())
