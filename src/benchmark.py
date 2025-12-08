#!/usr/bin/env python3
"""
Main benchmark orchestrator script.
Runs GreenMIR, Epoch and Folder-based benchmarks and generates all plots/tables.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"\nFAILED: {description}")
        return False
    
    print(f"\nSUCCESS: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks (GreenMIR + Epoch + Folder) and generate plots/tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything with default paths
  python src/benchmark.py --greenmir-outputs outputs/greenmir --epoch-outputs outputs/epoch
  
  # Run only GreenMIR benchmark
  python src/benchmark.py --greenmir-outputs outputs/greenmir --greenmir-only
  
  # Run only Epoch benchmark
  python src/benchmark.py --epoch-outputs outputs/epoch --epoch-only
  
  # Run only Folder-based benchmark (custom dataset from make_table_from_folder.py)
  python src/benchmark.py --folder-db example/test.db --folder-outputs outputs/folder --folder-only
  
  # Regenerate only plots (no benchmarks)
  python src/benchmark.py --plots-only
        """
    )
    
    # Input paths
    parser.add_argument("--greenmir-outputs", type=str,
                        help="Path to GreenMIR inference outputs (JSON files or directory)")
    parser.add_argument("--epoch-outputs", type=str,
                        help="Path to Epoch inference outputs (JSON file or directory)")
    parser.add_argument("--folder-outputs", type=str,
                        help="Path to Folder-based inference outputs (JSON file or directory)")
    
    # Database/dataset paths (with defaults)
    parser.add_argument("--greenmir-dataset", type=str, default="static/dataset.csv",
                        help="Path to GreenMIR dataset CSV (default: static/dataset.csv)")
    parser.add_argument("--greenmir-db", type=str, default="data/greenmir.db",
                        help="Path to greenmir.db (default: data/greenmir.db)")
    parser.add_argument("--epoch-db", type=str, default="data/epoch.db",
                        help="Path to epoch.db (default: data/epoch.db)")
    parser.add_argument("--folder-db", type=str,
                        help="Path to folder-based database (created by make_table_from_folder.py)")
    
    # Output directories (with defaults)
    parser.add_argument("--greenmir-results", type=str, default="results_id",
                        help="GreenMIR results directory (default: results_id)")
    parser.add_argument("--epoch-results", type=str, default="results_epoch",
                        help="Epoch results directory (default: results_epoch)")
    parser.add_argument("--folder-results", type=str, default="results_folder",
                        help="Folder-based results directory (default: results_folder)")
    
    # Control flags
    parser.add_argument("--greenmir-only", action="store_true",
                        help="Run only GreenMIR benchmark (skip Epoch and Folder)")
    parser.add_argument("--epoch-only", action="store_true",
                        help="Run only Epoch benchmark (skip GreenMIR and Folder)")
    parser.add_argument("--folder-only", action="store_true",
                        help="Run only Folder-based benchmark (skip GreenMIR and Epoch)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Only regenerate plots (skip benchmarks)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Run benchmarks but skip plot generation")
    
    args = parser.parse_args()
    
    # Determine which benchmarks to run
    run_greenmir = not args.epoch_only and not args.folder_only
    run_epoch = not args.greenmir_only and not args.folder_only
    run_folder = not args.greenmir_only and not args.epoch_only
    
    # If folder-only is set, only run folder
    if args.folder_only:
        run_greenmir = False
        run_epoch = False
        run_folder = True
    
    # Validation
    if not args.plots_only:
        if run_greenmir and not args.greenmir_outputs:
            print("Error: --greenmir-outputs is required (unless --epoch-only, --folder-only or --plots-only)")
            sys.exit(1)
        if run_epoch and not args.epoch_outputs:
            print("Error: --epoch-outputs is required (unless --greenmir-only, --folder-only or --plots-only)")
            sys.exit(1)
        if run_folder and args.folder_outputs and not args.folder_db:
            print("Error: --folder-db is required when using --folder-outputs")
            sys.exit(1)
    
    success = True
    
    # ========================================================================
    # BENCHMARKS
    # ========================================================================
    
    if not args.plots_only:
        # GreenMIR Benchmark
        if run_greenmir and args.greenmir_outputs:
            cmd = [
                "python", "src/benchmarks/benchmark_greenmir_id.py",
                "--dataset", args.greenmir_dataset,
                "--outputs-dir", args.greenmir_outputs,
                "--out-dir", args.greenmir_results,
                "--db", args.greenmir_db,
            ]
            success = run_command(cmd, "GreenMIR Benchmark") and success
        
        # Epoch Benchmark
        if run_epoch and args.epoch_outputs:
            cmd = [
                "python", "src/benchmarks/benchmark_epoch_id.py",
                "--database", args.epoch_db,
                "--outputs", args.epoch_outputs,
                "--out-dir", args.epoch_results,
                "--arxiv-only",
            ]
            success = run_command(cmd, "Epoch Benchmark (arXiv only)") and success
        
        # Folder-based Benchmark
        if run_folder and args.folder_outputs and args.folder_db:
            cmd = [
                "python", "src/benchmarks/benchmark_folder_id.py",
                "--database", args.folder_db,
                "--outputs", args.folder_outputs,
                "--out-dir", args.folder_results,
            ]
            success = run_command(cmd, "Folder-based Benchmark") and success
    
    # ========================================================================
    # PLOTS & TABLES
    # ========================================================================
    
    if not args.skip_plots:
        # GreenMIR filtered plots
        if run_greenmir and args.greenmir_outputs:
            cmd = [
                "python", "src/benchmarks/generate_filtered_plots.py",
                "--results-dir", args.greenmir_results,
            ]
            success = run_command(cmd, "GreenMIR Filtered Plots") and success
        
        # GreenMIR best table
        if run_greenmir and args.greenmir_outputs:
            cmd = [
                "python", "src/benchmarks/make_best_table.py",
                "--best",
            ]
            success = run_command(cmd, "GreenMIR Best Table") and success
        
        # Epoch plots
        if run_epoch and args.epoch_outputs:
            cmd = [
                "python", "src/benchmarks/generate_epoch_plots.py",
                "--result-dir", f"{args.epoch_results}/arxiv_only",
            ]
            success = run_command(cmd, "Epoch Plots") and success
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    if success:
        print("ALL BENCHMARKS AND PLOTS COMPLETED SUCCESSFULLY!")
    else:
        print("SOME STEPS FAILED - Check logs above")
    print(f"{'='*80}\n")
    
    # Show output locations
    if run_greenmir and args.greenmir_outputs:
        print(f"GreenMIR results: {args.greenmir_results}/")
    if run_epoch and args.epoch_outputs:
        print(f"Epoch results: {args.epoch_results}/")
    if run_folder and args.folder_outputs:
        print(f"Folder results: {args.folder_results}/")
    print()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
