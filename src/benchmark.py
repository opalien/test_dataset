#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print(f"\nFAILED: {description}")
        return False

    print(f"\nSUCCESS: {description}")
    return True


def run_greenmir(args: argparse.Namespace) -> int:
    success = True

    if not args.plots_only:
        cmd = [
            "python", "src/benchmarks/benchmark_greenmir_id.py",
            "--dataset", args.dataset,
            "--outputs-dir", args.outputs,
            "--out-dir", args.results,
            "--db", args.db,
        ]
        success = run_command(cmd, "GreenMIR Benchmark") and success

    if not args.skip_plots:
        cmd = ["python", "src/benchmarks/generate_filtered_plots.py", "--results-dir", args.results]
        success = run_command(cmd, "GreenMIR Filtered Plots") and success

        cmd = ["python", "src/benchmarks/make_best_table.py", "--best"]
        success = run_command(cmd, "GreenMIR Best Table") and success

    return 0 if success else 1


def run_epoch(args: argparse.Namespace) -> int:
    success = True

    if not args.plots_only:
        cmd = [
            "python", "src/benchmarks/benchmark_epoch_id.py",
            "--database", args.db,
            "--outputs", args.outputs,
            "--out-dir", args.results,
        ]
        if args.arxiv_only:
            cmd.append("--arxiv-only")
        success = run_command(cmd, "Epoch Benchmark") and success

    if not args.skip_plots:
        result_subdir = f"{args.results}/arxiv_only" if args.arxiv_only else args.results
        cmd = ["python", "src/benchmarks/generate_epoch_plots.py", "--result-dir", result_subdir]
        success = run_command(cmd, "Epoch Plots") and success

    return 0 if success else 1


def run_folder(args: argparse.Namespace) -> int:
    success = True

    if not args.plots_only:
        cmd = [
            "python", "src/benchmarks/benchmark_folder_id.py",
            "--database", args.db,
            "--outputs", args.outputs,
            "--out-dir", args.results,
        ]
        success = run_command(cmd, "Folder-based Benchmark") and success

    return 0 if success else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmarks and generate plots/tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/benchmark.py greenmir --outputs outputs/greenmir --db data/greenmir.db
  python src/benchmark.py epoch --outputs outputs/epoch --db data/epoch.db --arxiv-only
  python src/benchmark.py folder --outputs outputs/folder --db example/test.db
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    greenmir = subparsers.add_parser("greenmir", help="Run GreenMIR benchmark")
    greenmir.add_argument("--outputs", required=True)
    greenmir.add_argument("--db", default="data/greenmir.db")
    greenmir.add_argument("--dataset", default="static/dataset.csv")
    greenmir.add_argument("--results", default="results_id")
    greenmir.add_argument("--plots-only", action="store_true")
    greenmir.add_argument("--skip-plots", action="store_true")
    greenmir.set_defaults(func=run_greenmir)

    epoch = subparsers.add_parser("epoch", help="Run Epoch benchmark")
    epoch.add_argument("--outputs", required=True)
    epoch.add_argument("--db", default="data/epoch.db")
    epoch.add_argument("--results", default="results_epoch")
    epoch.add_argument("--arxiv-only", action="store_true")
    epoch.add_argument("--plots-only", action="store_true")
    epoch.add_argument("--skip-plots", action="store_true")
    epoch.set_defaults(func=run_epoch)

    folder = subparsers.add_parser("folder", help="Run folder-based benchmark")
    folder.add_argument("--outputs", required=True)
    folder.add_argument("--db", required=True)
    folder.add_argument("--results", default="results_folder")
    folder.add_argument("--plots-only", action="store_true")
    folder.add_argument("--skip-plots", action="store_true")
    folder.set_defaults(func=run_folder)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
