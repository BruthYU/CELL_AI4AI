import argparse as ap
import importlib.metadata
import logging

logger = logging.getLogger(__name__)


def parse_args_score(parser: ap.ArgumentParser):
    parser.add_argument(
        "-i",
        "--user-input",
        type=str,
        help="Path to aggregated results for user model",
        required=True,
    )
    parser.add_argument(
        "-I",
        "--base-input",
        type=str,
        help="Path to aggregated results for base model",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./baseline_diff.csv",
        help="Path to csv to write [default: %(default)s]",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def run_score(args: ap.Namespace):
    from .. import score_agg_metrics

    score_agg_metrics(
        results_user=args.user_input,
        results_base=args.base_input,
        output=args.output,
    )
