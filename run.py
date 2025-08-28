from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils.misc import get_frames, run_experiment
from utils.args import get_args
# from dataloader import build_data
from modeling import modeling_funcs_builder
from dataloader import data_builder



     


if __name__=='__main__':
    args = get_args()
    # dataset = build_data(args.dataset_name, args.video_root, args.data_files,  shuffle_video=args.shuffle_video, num_gpus=args.num_gpus, cur_gpu=args.cur_gpu, limit=args.limit, num_extra_video=args.num_extra_video)
    # dataset = get_dataset(args)
    dataset = data_builder.create_loader(args.dataset_name, args.video_root, args.data_files, args.shuffle_video, args.num_gpus, args.cur_gpu, args.limit, args.num_extra_video, args.use_local_parquest)
    print(f"dataset_name: {args.dataset_name}")
    num_gpus = args.num_gpus
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    assert gpu_count == num_gpus
    
    setup_model_func, get_inputs_func, decode_func = modeling_funcs_builder.get_modeling_funcs(args.model_base)


    run_experiment(
        get_inputs_func=get_inputs_func,
        decode_func=decode_func,
        dataset=dataset, 
        setup_model_func=setup_model_func,
        model_base=args.model_base, 
        device=f'cuda:{args.cur_gpu}', 
        result_dir=args.result_dir,
        shuffle_video=args.shuffle_video,
        shuffle_frame=args.shuffle_frame,
        frozen_video=args.frozen_video,
        no_video=args.no_video,
        num_gpus=args.num_gpus,
        cur_id=args.cur_gpu,
        combine_type=args.combine_type,
        custom_question=args.custom_question,
        add_extra_options=args.add_extra_options,
        no_target_video=args.no_target_video,
        replace_correct_with_extra=args.replace_correct_with_extra,
        max_num_frames=args.max_num_frames,
        max_new_tokens=args.max_new_tokens,
    )
