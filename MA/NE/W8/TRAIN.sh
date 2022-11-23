#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=w8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH -o /home/woody/iwi5/%u/Inp_Mono/Swwin/nbhd_emb/w8/window/w8-%j.out
#SBATCH -e /home/woody/iwi5/%u/Inp_Mono/Swwin/nbhd_emb/w8/window/w8-%j.err
#SBATCH --time=24:00:00
#Timelimit format: "hours:minutes:seconds"

# Tell pipenv to install the virtualenvs in the cluster folder
# export WORKON_HOME==/cluster/`whoami`/.python_cache
# export XDG_CACHE_DIR=/cluster/`whoami`/.cache
# export PYTHONUSERBASE=/cluster/`whoami`/.python_packages
# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)

mkdir /scratch/$SLURM_JOB_ID

mkdir /scratch/$SLURM_JOB_ID/train
mkdir /scratch/$SLURM_JOB_ID/val

tar -xf /home/woody/iwi5/iwi5069h/Mono/train/Legs.tar.gz -C /scratch/$SLURM_JOB_ID/train/
tar -xf /home/woody/iwi5/iwi5069h/Mono/train/Metal.tar.gz -C /scratch/$SLURM_JOB_ID/train/
tar -xf /home/woody/iwi5/iwi5069h/Mono/val/Legs.tar.gz -C /scratch/$SLURM_JOB_ID/val/
tar -xf /home/woody/iwi5/iwi5069h/Mono/val/Metal.tar.gz -C /scratch/$SLURM_JOB_ID/val/

scp /scratch/$SLURM_JOB_ID/train/Legs/projection6_1.tif /home/woody/iwi5/iwi5069h/Check/

#export CUDA_VISIBLE_DEVICES=1
python3 train.py --job_id=$SLURM_JOB_ID

rm -r /scratch/$SLURM_JOB_ID
