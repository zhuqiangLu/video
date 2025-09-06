from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils.misc import run_experiment
from utils.args import get_args
# from dataloader import build_data
from modeling import modeling_funcs_builder
from dataloader import data_builder
from exp_configurator.configurator import configurator

from transformers import logging
logging.set_verbosity_error()
     


if __name__=='__main__':
    args = get_args()

    # dataset = data_builder.create_loader(args.dataset_name, args.video_root, args.data_files, args.shuffle_video, args.num_gpus, args.cur_gpu, args.limit, args.num_extra_video, args.use_local_parquest, args.backend, args.max_num_frames)
    dataset = data_builder.create_loader(**vars(args))
    print(f"dataset_name: {args.dataset_name}")
    num_gpus = args.num_gpus
    gpu_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {gpu_count}")
    # assert gpu_count == num_gpus
    
    setup_model_func, get_inputs_func, inference_func = modeling_funcs_builder.get_modeling_funcs(args.model_base)

    exp_configurator = configurator(**vars(args))

    run_experiment(
        exp_configurator=exp_configurator,
        get_inputs_func=get_inputs_func,
        inference_func=inference_func,
        dataset=dataset, 
        setup_model_func=setup_model_func,
        device=f'cuda:{args.cur_gpu}',
        **vars(args)
    )
