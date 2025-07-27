import argparse


def build_parser():
    """Build parser to allow configuration of the dictionary parameters via command line arguments."""
    parser = argparse.ArgumentParser(description="Configure experiment parameters.")

    # Add arguments for each key in the dictionary 'c'
    parser.add_argument(
        '--exp_optimizer', type=str, default='lookahead_lamb',
        choices=['default', 'lamb', 'lookahead_lamb'],
        help="Optimizer type. Choose from 'default' (Adam), 'lamb', or 'lookahead_lamb'."
    )
    parser.add_argument(
        '--exp_lr', type=float, default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        '--exp_weight_decay', type=float, default=1e-5,
        help="Weight decay (L2 penalty)."
    )
    parser.add_argument(
        '--exp_lookahead_update_cadence', type=int, default=6,
        help="Lookahead update cadence (number of steps before updating slow weights)."
    )
    parser.add_argument(
        '--exp_scheduler', type=str, default='flat_and_anneal',
        help='Learning rate scheduler: see npt/optim.py for options.')
    parser.add_argument(
        '--exp_num_total_steps', type=float, default=200,
        help='Number of total gradient descent steps. The maximum number of '
             'epochs is computed as necessary using this value (e.g. in '
             'gradient syncing across data parallel replicates in distributed '
             'training).')
    parser.add_argument(
        '--exp_optimizer_warmup_proportion', type=float, default=0.7,
        help='The proportion of total steps over which we warmup.'
             'If this value is set to -1, we warmup for a fixed number of '
             'steps. Literature such as Evolved Transformer (So et al. 2019) '
             'warms up for 10K fixed steps, and decays for the rest. Can '
             'also be used in certain situations to determine tradeoff '
             'annealing, see exp_tradeoff_annealing_proportion below.')
    parser.add_argument(
        '--exp_batch_size', type=int, default=-1,
        help='Number of instances (rows) in each batch '
             'taken as input by the model. -1 corresponds to no '
             'minibatching.')
    # Parse the arguments
    args = parser.parse_args()

    # Return parsed arguments as a dictionary
    return parser


if __name__ == '__main__':
    # Get the configuration by calling the build_parser function
    c = build_parser()

    # You can now use `c` in the rest of your program
    print(c)
