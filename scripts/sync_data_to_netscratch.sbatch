#!/bin/sh
#SBATCH --job-name=sync_data
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH -t 0-12:00
#SBATCH -c 4
#SBATCH --output=/n/home04/rrinberg/catered_out/sync_data_%j.out
#SBATCH --error=/n/home04/rrinberg/catered_out/sync_data_%j.err

echo "Starting rsync from home to netscratch..."
echo "Source: /n/home04/rrinberg/data_dir/compression/"
echo "Dest:   /n/netscratch/sham_lab/Lab/rrinberg/compression/"
echo "========================================"

rsync -aP /n/home04/rrinberg/data_dir/compression/ /n/netscratch/sham_lab/Lab/rrinberg/compression/

echo "========================================"
echo "Done!"
