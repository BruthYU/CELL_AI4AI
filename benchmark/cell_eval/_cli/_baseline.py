import argparse as ap
import importlib.metadata
import logging

from ._const import DEFAULT_COUNTS_COL, DEFAULT_CTRL, DEFAULT_PERT_COL

logger = logging.getLogger(__name__)


def parse_args_baseline(parser: ap.ArgumentParser):
    """
    CLI for evaluation
    """
    parser.add_argument(
        "-a",
        "--adata",
        type=str,
        help="Path to the anndata",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--counts",
        type=str,
        help="Path to the perturbation cell counts",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path to save the baseline anndata [default: %(default)s]",
        default="./baseline.h5ad",
    )
    parser.add_argument(
        "-O",
        "--output-de-path",
        type=str,
        help="Path to save the baseline differential expression table [default: %(default)s]",
        default="./baseline_de.csv",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default=DEFAULT_CTRL,
        help="Name of the control perturbation [default: %(default)s]",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default=DEFAULT_PERT_COL,
        help="Name of the column designated perturbations [default: %(default)s]",
    )
    parser.add_argument(
        "--counts-col",
        type=str,
        default=DEFAULT_COUNTS_COL,
        help="Name of the column designated counts in input csv (if provided) [default: %(default)s]",
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use [default: %(default)s]",
    )
    parser.add_argument(
        "--allow-discrete",
        action="store_true",
        help="Bypass log normalization in case we incorrectly guess the data is discrete",
    )
    parser.add_argument(
        "--skip-de",
        action="store_true",
        help="Whether to skip differential expression analysis",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(
            version=importlib.metadata.version("cell_eval")
        ),
    )


def run_baseline(args: ap.Namespace):
    from .. import build_base_mean_adata

    pdex_kwargs = {
        "clip_value": 2**20,
        "is_log1p": True,  # Enforces log normalization internally
    }
    build_base_mean_adata(
        adata=args.adata,
        counts_df=args.counts,
        control_pert=args.control_pert,
        pert_col=args.pert_col,
        counts_col=args.counts_col,
        output_path=args.output_path,
        output_de_path=args.output_de_path if not args.skip_de else None,
        num_threads=args.num_threads,
        allow_discrete=args.allow_discrete,
        pdex_kwargs=pdex_kwargs,
    )
