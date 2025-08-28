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



TASK=TIGER-Lab/VideoEval-Pro
TASK=lmms-lab/Video-MME
# TASK=motionbench
RESULT_DIR=results_debug/test/$TASK


case $TASK in
    "lmms-lab/Video-MME")
        VIDEO_ROOT=/home/bingxing2/ailab/scxlab0109/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/data
        DATA_FILES=/home/bingxing2/ailab/scxlab0109/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/videomme/test-00000-of-00001.parquet
        ;;
    "TIGER-Lab/VideoEval-Pro")
        VIDEO_ROOT=/home/bingxing2/ailab/scxlab0109/.cache/huggingface/hub/datasets--TIGER-Lab--VideoEval-Pro/snapshots/a38a853b22576c7918e9cc0d1d4eedf9d46a1cae/videos/videos_filtered/
        DATA_FILES=/home/bingxing2/ailab/scxlab0109/.cache/huggingface/hub/datasets--TIGER-Lab--VideoEval-Pro/snapshots/a38a853b22576c7918e9cc0d1d4eedf9d46a1cae/data/test-00000-of-00001.parquet
        ;;
    "motionbench")
        VIDEO_ROOT=/home/bingxing2/ailab/scxlab0109/.cache/huggingface/hub/datasets--zhuqiang--motion_data/snapshots/2d0613058c3f98330061bfad717d39b832f65f8f/MotionBench/
        DATA_FILES=None
        ;;
    *)
        echo "Invalid task"
        exit 1
        ;;
esac

# MODEL_BASE=models/gemma-3n-E2B-it
# MODEL_BASE=models/Phi-3.5-vision-instruct
# MODEL_BASE=models/Qwen2.5-VL-3B-Instruct
# MODEL_BASE=models/SmolVLM2-2.2B-Instruct
# MODEL_BASE=models/InternVL3_5-2B

MODEL_BASE=google/gemma-3n-E2B-it
MODEL_BASE=microsoft/Phi-3.5-vision-instruct
MODEL_BASE=Qwen/Qwen2.5-VL-3B-Instruct
MODEL_BASE=HuggingFaceTB/SmolVLM2-2.2B-Instruct
# MODEL_BASE=models/InternVL3_5-2B

# NUM_GPUS=2
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
        --max_num_frames 16 \
        --shuffle_frame 
        --limit 1.0 \
        --use_local_parquest \
        --cur_gpu $i &
done
wait
