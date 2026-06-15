#!/bin/bash
#SBATCH --job-name=hf_upload
#SBATCH --partition=seas_compute,shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/n/home04/rrinberg/catered_out/hf_upload_%j.out

PYTHON=/n/holylabs/vadhan_lab/Lab/rrinberg/envs/conda/envs/MATS-inference2/bin/python

cd /n/home04/rrinberg/code/MATS/model-compression-bit-limiting

$PYTHON scripts/upload_loras_to_hf.py
