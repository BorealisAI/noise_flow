# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str,
                        default='./logdir/', help="Location to save logs")
    parser.add_argument("--sidd_path", type=str,
                        default='./data/SIDD_Medium_Raw/Data', help="Location of the SIDD dataset")
    parser.add_argument("--problem", type=str, default='sidd',
                        help="mnist | cifar10 | celeba | sidd")
    parser.add_argument("--dal", type=int, default=0,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int,
                        default=-1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=128, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=100, help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--optim", type=str, default='adam', help="sgd | adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default='glow',
                        help="glow")
    parser.add_argument("--decomp", type=str, default='LU',
                        help="Linear Decomposition Type (NONE, LU or SVD)")
    parser.add_argument("--full-conv", type=int, default=1,
                        help="Use full convolutions (vs sandwich)")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=8,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=10,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=1,
                        help="Number of levels")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")
    parser.add_argument("--squeeze-type", type=str, default='chessboard',
                        help="Squeeze2d Type (path or chessboard)")
    parser.add_argument("--squeeze_factor", type=int, default=1,
                        help="Squeeze2d factor (1, 2, 4, ...)")
    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=1,
                        help="Type of flow. 0=reverse, 1=1x1conv")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    # for SIDD
    parser.add_argument("--noise_baseline", type=str, default='gauss',
                        help="Noise synthesis baseline: gauss | nlf")
    parser.add_argument("--cond_gt", action='store_true',
                        help="Conditional flow on ground truth (clean) image")
    parser.add_argument("--patch_height", type=int,
                        help="Patch height, width will be the same")
    parser.add_argument("--patch_sampling", type=str,
                        help="Patch sampling method form full images (uniform | random)")
    parser.add_argument("--n_tr_inst", type=int,
                        help="Number of training scene instances")
    parser.add_argument("--n_ts_inst", type=int,
                        help="Number of testing scene instances")
    parser.add_argument("--n_patches_per_image", type=int,
                        help="Max. number of patches sampled from each image")
    parser.add_argument("--reload_freq", type=int,  # TODO: check if this is still needed
                        help="Number of epochs to pass before reloading another set of training images")
    parser.add_argument("--server", type=str, default='skynet',
                        help="Compute server: skynet | eecs")
    parser.add_argument("--start_tr_im_idx", type=int,
                        help="Starting image index for training")
    parser.add_argument("--end_tr_im_idx", type=int,
                        help="Ending image index for training")
    parser.add_argument("--start_ts_im_idx", type=int,
                        help="Starting image index for testing")
    parser.add_argument("--end_ts_im_idx", type=int,
                        help="Ending image index for testing")
    parser.add_argument("--n_reuse_batch", type=int,
                        help="Number of times a minibatch is reused for training")  # to lower image loading frequency
    parser.add_argument("--n_reuse_image", type=int,
                        help="Number of times an image is reused for training")  # to lower image loading frequency
    parser.add_argument("--calc_pat_stats_and_baselines_only", action='store_true',
                        help="Calculate patch stats and baselines only")
    parser.add_argument("--split_observations", action='store_true',
                        help="Split noise observations between training and testing")
    parser.add_argument("--calc_hists_only", action='store_true',
                        help="Calculate histograms of train/test sets")
    parser.add_argument("--camera", type=str,
                        help="To choose training scenes from one camera")  # to lower image loading frequency
    parser.add_argument("--iso", type=int,
                        help="To choose training scenes from one camera")  # to lower image loading frequency
    parser.add_argument("--num_gpus", type=int,
                        help="Number of GPUs used for training")
    parser.add_argument("--sidd_cond", type=str, default='mix',
                        help="Type of architecture (e.g., uncond=Unconditional, mix=Mixture, etc.")
    parser.add_argument("--init_sdn", action='store_true',
                        help="Initialize affine coupling layers with signal-dependent model")
    parser.add_argument("--visualize", action='store_true',
                        help="Visualize sampled images")
    parser.add_argument("--append_sdn", action='store_true',
                        help="append signal-dependent noise fitting layer of degree 3")
    parser.add_argument("--append_sdn2", action='store_true',
                        help="append signal-dependent noise fitting layer of degree 2")
    parser.add_argument("--append_sdn_first", action='store_true',
                        help="append signal-dependent noise fitting layer next to base measure")
    parser.add_argument("--append_cY", action='store_true',
                        help="append AffineCouplingCondY layer before AffineCoupling layers")
    parser.add_argument("--arch", type=str, default='',
                        help="Defines a mixture architecture of bijectors")
    parser.add_argument("--continue_training", action='store_true',
                        help="To continue training be loading last trained model, if any.")
    parser.add_argument("--visualize_only", action='store_true',
                        help="To only visualize sampled noisy images.")
    parser.add_argument("--do_sample", action='store_true',
                        help="To sample noisy images from the test set.")
    parser.add_argument("--sample_epoch_start", type=int,
                        help="Starting epoch checkpoint to use for sampling")
    parser.add_argument("--sample_epoch_end", type=int,
                        help="Ending epoch checkpoint to use for sampling")
    parser.add_argument("--vis_samples", action='store_true',
                        help="Whether to visualize samples.")
    parser.add_argument("--copy_stats", action='store_true',
                        help="Whether to copy stats and baselines from parent directory.")
    parser.add_argument("--visualize_last_epoch", type=int,
                        help="Last epcoh visualized")  # to lower image loading frequency
    parser.add_argument("--sample_subdir", type=str,
                        help="Name of samples subdirectory")
    parser.add_argument("--temp", type=float,
                        help="Sampling temperature")  # to lower image loading frequency
    parser.add_argument("--save_batches", action='store_true',
                        help="Whether to save mini batches for faster loading later.")
    parser.add_argument("--load_batches", action='store_true',
                        help="Whether to load mini batches instead of sampling them.")
    parser.add_argument("--n_train_threads", type=int,
                        help="Number of training/testing threads")
    parser.add_argument("--fcsize", type=int,
                        help="Size of Fourier convolutional filters")
    parser.add_argument("--mb_qsize", type=int,
                        help="Size of minibatch queue")
    parser.add_argument("--mb_requeue", action='store_true',
                        help="Whether to requeue minibatches.")
    parser.add_argument("--shuffle_patches", action='store_true',
                        help="Whether to shuffle patches while loading.")
    parser.add_argument("--collect_vars", action='store_true',
                        help="Whether to collect variables from trained models.")
    parser.add_argument("--gain_init", type=float, default=-5.0,
                        help="An initial value for gain parameters")
    parser.add_argument("--pre_init", action='store_true',
                        help="Whether to initialize signal, gain, and/or camera variables from prior models.")
    hps = parser.parse_args()  # So error if typo
    return hps
