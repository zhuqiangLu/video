
# config.sh

# TASK=TIGER-Lab/VideoEval-Pro
# TASK=lmms-lab/Video-MME
TASK=motionbench
# TASK=mvbench
PPL=False

RESULT_DIR=results/acc/$TASK
if [ "$PPL" = True ]; then
	RESULT_DIR="results/ppl/$TASK"
else
	RESULT_DIR="results/acc/$TASK"
fi

echo $RESULT_DIR


BACKEND=av
LIMIT=1.0
MAX_NEW_TOKENS=128
export HF_ENDPOINT=https://hf-mirror.com


export http_proxy=https://wanglintao:iHiz3wPzlBGIJ2YEkSlVlG8GKoffQNgbQVv9S6SswYQw1tr3Lvf7N7tTwGCW@blsc-proxy.pjlab.org.cn:13128
export https_proxy=https://wanglintao:iHiz3wPzlBGIJ2YEkSlVlG8GKoffQNgbQVv9S6SswYQw1tr3Lvf7N7tTwGCW@blsc-proxy.pjlab.org.cn:13128


# MODEL_BASE=HuggingFaceTB/SmolVLM2-2.2B-Instruct
# NUM_FRAMES=16

# MODEL_BASE=microsoft/Phi-3.5-vision-instruct
# NUM_FRAMES=16

# MODEL_BASE=llava-hf/llava-onevision-qwen2-0.5b-ov-hf
# NUM_FRAMES=8

# MODEL_BASE=OpenGVLab/InternVL3_5-2B
# NUM_FRAMES=16

# MODEL_BASE=models/Qwen2.5-VL-3B-Instruct
MODEL_BASE=models/Qwen2.5-VL-7B-Instruct
MODEL_BASE=models/Video-R1-7B
# MODEL_BASE=checkpoints/test_run_Qwen2.5-VL
NUM_FRAMES=16



# MODEL_BASE=google/gemma-3n-E2B-it
# NUM_FRAMES=16


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
        VIDEO_ROOT=benchmarks/motionbench/MotionBench/
        DATA_FILES=None
        ;;
    "mvbench")
        VIDEO_ROOT=/home/bingxing2/ailab/scxlab0109/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots/230a2d4fac8900333c61754641c7a13e069ac9c6/video/
        DATA_FILES=/home/bingxing2/ailab/scxlab0109/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots/230a2d4fac8900333c61754641c7a13e069ac9c6/json/
        ;;
    *)
        echo "Invalid task"
        exit 1
        ;;
esac
