import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for this bench')
    parser.add_argument('--dataset_name', default='videomme', type=str, help='Specify the dataset.')
    parser.add_argument('--data_files', default=None, type=str, help='Specify the data files.')
    parser.add_argument('--video_root', default='', type=str, help='Specify the video root.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split.')
    parser.add_argument("--model_base", type=str, default="/path/to/qwen-model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--result_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPU device to use")
    parser.add_argument("--cur_gpu", type=int, default=0, help="Current GPU device")
    parser.add_argument("--no_video",  action="store_true", default=False, help="Not passing video to model")
    parser.add_argument("--shuffle_video", action="store_true", default=False, help="Shuffle video")
    parser.add_argument("--shuffle_frame", action="store_true", default=False, help="Shuffle frame")
    parser.add_argument("--reverse_frame", action="store_true", default=False, help="Reverse frame")
    parser.add_argument("--limit", type=float, default=1,  help="dataset size")
    parser.add_argument("--combine_type", type=str, default=None, help="combine type", choices=["target_first", "target_first", "target_middle"])
    parser.add_argument("--custom_question", type=str, default=None, help="custom question", choices=["video_position", "video_number", "count_frame", 'frozen_video_bool'])
    parser.add_argument("--add_extra_options", action="store_true", help="add extra options")
    parser.add_argument("--no_target_video", action="store_true", help="no target video")
    parser.add_argument("--replace_correct_with_extra", action="store_true",  default=False,help="replace correct with extra video")
    parser.add_argument("--num_extra_video", type=int, default=0, help="number of extra video")
    parser.add_argument("--max_num_frames", type=int, default=16, help="max number of frames")
    parser.add_argument("--use_local_parquest", action="store_true", default=False, help="use local parquest")
    parser.add_argument("--frozen_video", action="store_true", default=False, help="frozen video")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="max new tokens")
    parser.add_argument("--resume", action="store_true", default=False, help="resume")
    parser.add_argument("--backend", type=str, default="decord", help="video decoding backend")
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    parser.add_argument("--ppl", type=str2bool, nargs="?", const=True, default=False, help="Pass True/False")
    return parser.parse_args()

