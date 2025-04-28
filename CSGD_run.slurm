#!/bin/bash

#SBATCH -J TACT
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -o TACT.log
#SBATCH -e TACT_err.log
#SBATCH --gres=gpu:1

source activate TACT

cd code/Ranking
bash run_once.sh WN18RR_v1 8 8 2 0.01 0.01 demo 10 0
bash run_once.sh WN18RR_v2 8 8 2 0.01 0.01 demo 10 0
bash run_once.sh WN18RR_v3 8 8 2 0.01 0.01 demo 10 0
bash run_once.sh WN18RR_v4 8 8 2 0.01 0.01 demo 10 0

bash run_once.sh fb237_v1 16 8 2 0.005 0.01 demo 10 0
bash run_once.sh fb237_v2 16 8 2 0.005 0.01 demo 10 0
bash run_once.sh fb237_v3 16 8 2 0.005 0.01 demo 10 0
bash run_once.sh fb237_v4 16 8 2 0.005 0.01 demo 10 0
