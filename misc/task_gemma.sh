
TASK=TIGER-Lab/VideoEval-Pro
TASK=lmms-lab/Video-MME
# TASK=motionbench
RESULT_DIR=results_debug/test/$TASK


case $TASK in
    "lmms-lab/Video-MME")
        VIDEO_ROOT=benchmarks/videomme/data
        DATA_FILES=benchmarks/videomme/test-00000-of-00001.parquet
        ;;
    "TIGER-Lab/VideoEval-Pro")
        VIDEO_ROOT=videoevalpro/videos/videos_filtered/
        DATA_FILES=videoevalpro/data/test-00000-of-00001.parquet
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

MODEL_BASE=models/gemma-3n-E2B-it
MODEL_BASE=models/Phi-3.5-vision-instruct

NUM_GPUS=2
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
        --limit 1.0 \
        --use_local_parquest \
        --cur_gpu $i &
done
wait
