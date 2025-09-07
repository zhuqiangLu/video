#!/bin/bash
module load anaconda/2021.11
export CUDA_HOME=/home/bingxing2/apps/cuda/11.7.0
module load ffmpeg/4.4.1-gcc11  
module load anaconda/2021.11 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x compilers/gcc/11.3.0
conda activate infinity2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export TRANSFORMERS_VERBOSITY="info"

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

export http_proxy=https://wanglintao:MEKLD6LPq5BPPDtozP0ag4ErMiYnOYwtoWKkqgjsLgmqlz4JsIqQGIaCxrST@blsc-proxy.pjlab.org.cn:13128
export https_proxy=https://wanglintao:MEKLD6LPq5BPPDtozP0ag4ErMiYnOYwtoWKkqgjsLgmqlz4JsIqQGIaCxrST@blsc-proxy.pjlab.org.cn:13128


# nodes
NODES=$1

# gpus
NUM_GPUS=$2

# rank
NODE_RANK=$3

# Master addr
MASTER_ADDR=$4
MASTER_PORT=29501

#DHOSTFILE
DHOSTFILE=$5

# JOB_ID
JOB_ID=$6

# logs
OUTPUT_LOG="./log/train_rank${NODE_RANK}_${JOB_ID}.log"


# export CUDA_HOME=/home/bingxing2/apps/cuda/11.7.0
export LD_PRELOAD=/home/bingxing2/ailab/scxlab0109/.conda/envs/dna_ft/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

source tasks/config.sh

echo $TASK
echo $VIDEO_ROOT
echo $DATA_FILES
echo $RESULT_DIR
echo $BACKEND
echo $MAX_NEW_TOKENS
echo $LIMIT

export HF_HUB_OFFLINE=1

MODEL_BASE=HuggingFaceTB/SmolVLM2-2.2B-Instruct

NUM_FRAMES=16
# DEFAULT 
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait

# shuffle frame
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --shuffle_frame \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait

# shuffle video
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --shuffle_video \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait

# frozen video
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --frozen_video \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait

# frozen video + custom question: bool
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --frozen_video \
        --custom_question "frozen_video_bool" \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait






#  no correct answer
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --replace_correct_with_extra \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait

#  no video
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --no_video \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait


# add extra options
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --add_extra_options \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --cur_gpu $i &
done
wait