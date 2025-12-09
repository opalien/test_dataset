#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))

from core.metrics import load_results_data, compute_f1_stats
from core.plotting import plot_gradient_stacked_filtered, create_f1_table


def process_directory(result_dir: Path, exclude: set[str] | None = None) -> None:
    print(f"Processing: {result_dir}")

    try:
        distances, missing = load_results_data(result_dir, exclude_fields=exclude)
    except FileNotFoundError as e:
        print(f"  Skip: {e}")
        return

    overview_dir = result_dir / "overview"
    overview_dir.mkdir(exist_ok=True)

    plot_gradient_stacked_filtered(overview_dir / "grad_stacked_filtered.png", distances, missing)
    print(f"  Saved: grad_stacked_filtered.png")

    stats = compute_f1_stats(distances, missing)
    create_f1_table(stats, overview_dir / "f1_table.png")
    print(f"  Saved: f1_table.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate filtered plots from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/plot.py --results results_id/greenmir_strict
  python src/plot.py --results results_epoch/all
  python src/plot.py --results results_folder/benchmark
  python src/plot.py --all-greenmir
  python src/plot.py --all-epoch
        """
    )
    parser.add_argument("--results", type=str, help="Single results directory to process")
    parser.add_argument("--all-greenmir", action="store_true", help="Process all greenmir_* in results_id/")
    parser.add_argument("--all-epoch", action="store_true", help="Process all subdirs in results_epoch/")
    parser.add_argument("--exclude", nargs="*", default=[], help="Fields to exclude from plots")
    args = parser.parse_args()

    exclude = set(args.exclude) if args.exclude else None

    if args.results:
        process_directory(Path(args.results), exclude)

    elif args.all_greenmir:
        base = Path("results_id")
        if not base.exists():
            print(f"Directory not found: {base}")
            return
        for d in sorted(base.glob("greenmir_*")):
            if d.is_dir():
                process_directory(d, exclude)

    elif args.all_epoch:
        base = Path("results_epoch")
        if not base.exists():
            print(f"Directory not found: {base}")
            return
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / "matched_models.csv").exists():
                process_directory(d, exclude)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
