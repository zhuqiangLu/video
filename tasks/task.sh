#!/bin/bash
source tasks/config.sh

echo $TASK
echo $VIDEO_ROOT
echo $DATA_FILES
echo $RESULT_DIR
echo $BACKEND
echo $MAX_NEW_TOKENS
echo $LIMIT
echo $MODEL_BASE

export HF_HUB_OFFLINE=1
NUM_GPUS=4


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
        --ppl $PPL \
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
        --ppl $PPL \
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
        --ppl $PPL \
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
        --ppl $PPL \
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
        --ppl $PPL \
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
        --ppl $PPL \
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
        --ppl $PPL \
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
        --ppl $PPL \
        --cur_gpu $i &
done
wait



# reverse frame
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --reverse_frame \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --ppl $PPL \
        --cur_gpu $i &
done
wait


# reverse frame
for i in $(seq 0 $((NUM_GPUS-1)))
do
    python run.py \
        --dataset_name $TASK \
        --video_root  $VIDEO_ROOT \
        --data_files $DATA_FILES \
        --split test \
        --model_base $MODEL_BASE \
        --batch_size 1 \
        --reverse_frame \
        --add_extra_options \
        --result_dir $RESULT_DIR \
        --num_gpus $NUM_GPUS \
        --max_num_frames $NUM_FRAMES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --limit $LIMIT \
        --use_local_parquest \
        --backend $BACKEND \
        --ppl $PPL \
        --cur_gpu $i &
done
wait
