"""Run one experiment from SPEC Section 14, e.g. --exp E2 --split random."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, choices=["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"])
    parser.add_argument("--split", choices=["random", "temporal"], default="random")
    parser.add_argument("--model", choices=["logreg", "forest", "hgb"],
                        help="model for E4-E7 (default: best validation catch rate among E1-E3)")
    args = parser.parse_args()
    result = experiment.run(args.exp, args.split, args.model)
    summary = {k: v for k, v in result.items() if k != "trials"} if isinstance(result, dict) else result
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
