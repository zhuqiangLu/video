from .videomme import VideoMMEDataset
from .videoevalpro import VideoEvalProDataset
from .motionbench import MotionBenchDataset
from .builder import data_builder
# __all__ = ['VideoMMEDataset', 'VideoEvalProDataset', 'MotionBenchDataset']





# def build_data(dataset_name, video_root, data_files=None, shuffle_video=False, num_gpus=1, cur_gpu=0, limit=None, num_extra_video=0, use_local_parquest=False):
#     if dataset_name == 'videomme' or 'Video-MME' in dataset_name:
#         return VideoMMEDataset(video_root, dataset_name, data_files, shuffle_video, num_gpus, cur_gpu, limit, num_extra_video, use_local_parquest)
#     elif dataset_name == 'videoevalpro' or 'VideoEval-Pro' in dataset_name:
#         return VideoEvalProDataset(video_root, dataset_name, data_files, shuffle_video, num_gpus, cur_gpu, limit, num_extra_video, use_local_parquest)
#     elif dataset_name == 'motionbench':
#         return MotionBenchDataset(video_root, dataset_name, data_files, shuffle_video, num_gpus, cur_gpu, limit, num_extra_video)
#     else:
#         raise ValueError(f"Invalid dataset name: {dataset_name}")