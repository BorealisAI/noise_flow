#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=logs/job_output_%j.out
#SBATCH --error=logs/job_error_%j.out
#SBATCH --job-name=dncnn_nf


hostname
whoami


### Training


## Train DnCNN with Gaussian noise
python train_dncnn_noiseflow.py \
        --model DnCNN_Gauss \
        --train_data '/home/abdo/Downloads/SIDD_Medium_Raw/Data' \
        --save_every 20 \
        --max_epoch 2000 \
        --num_gpus 1


## Train DnCNN with signal-dependent noise using camera noise level functions
#python train_dncnn_noiseflow.py \
#        --model DnCNN_CamNLF \
#        --train_data '/home/abdo/Downloads/SIDD_Medium_Raw/Data' \
#        --save_every 20 \
#        --max_epoch 2000 \
#        --num_gpus 1


## Train DnCNN with noise generated with Noise Flow
#python train_dncnn_noiseflow.py \
#        --model DnCNN_NF \
#        --train_data '/home/abdo/Downloads/SIDD_Medium_Raw/Data' \
#        --save_every 20 \
#        --max_epoch 2000 \
#        --num_gpus 1


## Train DnCNN with real noise
#python train_dncnn_noiseflow.py \
#        --model DnCNN_Real \
#        --train_data '/home/abdo/Downloads/SIDD_Medium_Raw/Data' \
#        --save_every 20 \
#        --max_epoch 2000 \
#        --num_gpus 1


### Testing


#python test_dncnn_noiseflow.py \
#        --model_name 'DnCNN_Gauss' \
#        --save_result \
#        --min_epc 1 \
#        --max_epc 2000 \
#        --epc_step 20


#python test_dncnn_noiseflow.py \
#        --model_name 'DnCNN_CamNLF' \
#        --save_result \
#        --min_epc 1 \
#        --max_epc 2000 \
#        --epc_step 20


#python test_dncnn_noiseflow.py \
#        --model_name 'DnCNN_NF' \
#        --save_result \
#        --min_epc 1 \
#        --max_epc 2000 \
#        --epc_step 20


#python test_dncnn_noiseflow.py \
#        --model_name 'DnCNN_Real' \
#        --save_result \
#        --min_epc 1 \
#        --max_epc 2000 \
#        --epc_step 20
