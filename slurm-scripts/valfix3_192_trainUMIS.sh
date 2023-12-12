#!/bin/bash

#$ -P papiez.prjc
#$ -N nn-unet

echo "------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

source /gpfs3/well/papiez/users/hri611/python/env1-${MODULE_CPU_TYPE}/bin/activate

python trainUMIS.py --train_file train_imgs.txt --val_file val_imgs.txt --test_file test_imgs.txt --experiment U_unc_oversample_192 --dataset autopet --Uncertainty_Loss True --crop_H 192 --crop_W 192 --crop_D 192 --end_epochs 400


#submit with: sbatch -p gpu_long --gres gpu:1 U_overssample_128_trainUMIS.sh
