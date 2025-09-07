
# config.sh

TASK=TIGER-Lab/VideoEval-Pro
TASK=lmms-lab/Video-MME
# TASK=motionbench
RESULT_DIR=results/98/$TASK
BACKEND=decord
LIMIT=1.0
MAX_NEW_TOKENS=128

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