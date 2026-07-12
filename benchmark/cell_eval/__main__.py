import argparse as ap
import logging

from ._cli import (
    parse_args_baseline,
    parse_args_prep,
    parse_args_run,
    parse_args_score,
    run_baseline,
    run_evaluation,
    run_prep,
    run_score,
)

logger = logging.getLogger(__name__)


def get_args():
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    parse_args_prep(subparsers.add_parser("prep"))
    parse_args_run(subparsers.add_parser("run"))
    parse_args_baseline(subparsers.add_parser("baseline"))
    parse_args_score(subparsers.add_parser("score"))
    return parser.parse_args()


def main():
    """
    Main function to run the evaluation.
    """
    args = get_args()
    match args.subcommand:
        case "prep":
            run_prep(args)
        case "run":
            run_evaluation(args)
        case "baseline":
            run_baseline(args)
        case "score":
            run_score(args)


if __name__ == "__main__":
    main()
