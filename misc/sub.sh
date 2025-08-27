#!/bin/sh

# For BJ Cluster
# export PYTHONUNBUFFERED=1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/miniconda3/lib:/home/bingxing2/apps/cudnn/8.4.0.27_cuda11.x/lib

echo $CUDA_HOME
n_node=${1:-4}
n_gpu=${2:-4}
script=${3:-"./rank_7b_ds.sh"}

# script=${2:-"./run_multi_nodes.sh"}


GROUP="vip_gpu_ailab"
# ACCOUNT="ai4multi"
# ACCOUNT="ai4bio"
GROUP="vip_gpu_ailab_low"
ACCOUNT="ailab"

sbatch -o ./log/slurm-%j.out -N ${n_node} --qos=gpugpu -p $GROUP -A $ACCOUNT --gres=gpu:${n_gpu} misc/run_multi_nodes.sh $n_gpu $script
