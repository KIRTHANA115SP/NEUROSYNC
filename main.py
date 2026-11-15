import argparse
from pathlib import Path

from src.demo import run_demo
from src.physionet import run_physionet


def main():
    parser = argparse.ArgumentParser(
        description="NEUROSYNC: EEG signal exploration starter"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true",
                       help="Run a synthetic EEG smoke test (no download)")
    group.add_argument("--physionet", action="store_true",
                       help="Download and explore a small real PhysioNet EEG sample")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    if args.demo:
        run_demo()
    elif args.physionet:
        run_physionet()


if __name__ == "__main__":
    main()
