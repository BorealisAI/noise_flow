#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=logs/job_output_%j.out
#SBATCH --error=logs/job_error_%j.out
#SBATCH --job-name=nf_test
# the above settings are for slurm (if applicable)

hostname
whoami

# make sure to feed in the path to SIDD dataset (full or medium)
# examples:
# --sidd_path '/shared-data/SIDD/'
# --sidd_path './data/SIDD_Medium_Raw/Data'
# --sidd_path '/local/ssd/kamel/SIDD_Medium_Raw_Dir/'

# --cam and --iso are optional, use them to train on a specific camera and/or a specific ISO level
# --width is the number of filters in each of the affine coupling layers (for Noise Flow it is 32)

# to try any of the following examples, uncomment then execute this file or submit it to a job server
# (e.g., using slurm), alternatively, you may just run the command from a terminal

#`--width`: number of filters in each affine coupling layers, if used
#`--n_batch_train`: training batch size
#`--n_batch_test`: testing batch size
#`--epochs`: number of training epochs
#`--lr`: learning rate




# Example: training the Noise Flow model (S-Ax4-G-Ax4-CAM), number of parameters 2433

python3 train_noise_flow.py --logdir noise_flow_model   --arch "sdn5|unc|unc|unc|unc|gain4|unc|unc|unc|unc" \
     --sidd_path './data/SIDD_Medium_Raw/Data' --n_train_threads 16   \
     --width 4 --epochs 2000  --lr 1e-4 --n_batch_train 138 --n_batch_test 138 --epochs_full_valid 10 \
     --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform \
     --start_tr_im_idx 10   --end_tr_im_idx 12   --start_ts_im_idx 10   --end_ts_im_idx 12


# Example: training a noise flow model with only affine coupling blocks of size 1 (S-Ax1-G-Ax1-CAM)

python3 train_noise_flow.py --logdir S-Ax1-G-Ax1-CAM   --arch "sdn5|unc|gain4|unc" \
     --sidd_path './data/SIDD_Medium_Raw/Data' --cam IP --iso 800  --n_train_threads 16   \
     --width 4 --epochs 2000  --lr 1e-4 --n_batch_train 138 --n_batch_test 138 --epochs_full_valid 10 \
     --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform \
     --start_tr_im_idx 10   --end_tr_im_idx 12   --start_ts_im_idx 10   --end_ts_im_idx 12


# Example: training a noise flow model with signal-dependent layer, gain layer, and camera parameters (S-G-CAM)

python3 train_noise_flow.py --logdir S-G-CAM   --arch "sdn5|gain4" \
     --sidd_path './data/SIDD_Medium_Raw/Data' --cam IP --iso 800 --n_train_threads 16   \
     --width 4 --epochs 2000  --lr 1e-4 --n_batch_train 138 --n_batch_test 138 --epochs_full_valid 10 \
     --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform \
     --start_tr_im_idx 10   --end_tr_im_idx 12   --start_ts_im_idx 10   --end_ts_im_idx 12


# Example: training a noise flow model with signal-dependent layer and gain layer, without camera parameters (S-G)

python3 train_noise_flow.py --logdir S-G   --arch "sdn4|gain4" \
     --sidd_path './data/SIDD_Medium_Raw/Data' --cam IP --iso 800 --n_train_threads 16   \
     --width 4 --epochs 2000  --lr 1e-4 --n_batch_train 138 --n_batch_test 138 --epochs_full_valid 10 \
     --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform \
     --start_tr_im_idx 10   --end_tr_im_idx 12   --start_ts_im_idx 10   --end_ts_im_idx 12


# Example: training an unconditional noise flow model of 4 affine coupling blocks (Ax4)

python3 train_noise_flow.py --logdir Ax4   --arch "unc|unc|unc|unc" \
     --sidd_path './data/SIDD_Medium_Raw/Data' --cam IP --iso 800 --n_train_threads 16   \
     --width 4 --epochs 2000  --lr 1e-4 --n_batch_train 138 --n_batch_test 138 --epochs_full_valid 10 \
     --n_patches_per_image 2898 --patch_height 32 --patch_sampling uniform \
     --start_tr_im_idx 10   --end_tr_im_idx 12   --start_ts_im_idx 10   --end_ts_im_idx 12
