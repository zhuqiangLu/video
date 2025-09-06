import os 
from datasets import load_dataset
from torch.utils.data import Dataset
import random
from .utils import split_data, sample_frames


from .builder import register


@register("VideoEval-Pro")
class VideoEvalProDataset(Dataset):
    def __init__(self, dataset_path, dataset_name='TIGER-Lab/VideoEval-Pro', data_files=None, shuffle_video=False, num_gpus=1, cur_gpu=0, limit=None, num_extra_video=0, use_local_parquest=False, backend='decord', max_num_frames=10):
        print(dataset_name, data_files)
        if use_local_parquest:
            self.dataset = load_dataset("parquet", split='test', data_files={'test': data_files})
        else:
            self.dataset = load_dataset(dataset_name, split='test', data_files={'test': data_files})

        self.video_root = os.path.join(dataset_path)
        self.shuffle_video = shuffle_video
        self.video_list = os.listdir(self.video_root)
        self.num_extra_video = num_extra_video
        self.dataset = split_data(self.dataset, num_gpus, limit)[cur_gpu]
        self.num_extra_video = num_extra_video
        self.backend = backend
        self.max_num_frames = max_num_frames
        # if limit is not None:
        #     self.dataset = self.dataset[:int(limit*len(self.dataset))]
            # print(len(self.dataset))



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.shuffle_video:
            print('shuffle video')
            video_path = random.choice(os.listdir(self.video_root))
            video_path = os.path.join(self.video_root, video_path)
        else:
            video_path = os.path.join(self.video_root, item['video'])

        extra_video_paths = [os.path.join(self.video_root, p) for p in random.sample(os.listdir(self.video_root), self.num_extra_video)]

        question = item['question']
        options = item['options']
        answer = item['answer']


        frames = sample_frames(video_path, max_num_frames=self.max_num_frames, backend=self.backend)
        extra_frames = list() 
        for extra_video_path in extra_video_paths:
            extra_frames.append(sample_frames(extra_video_path, max_num_frames=self.max_num_frames, backend=self.backend))

            

        data_item = {
            'frames': frames,
            'extra_frames': extra_frames,
            'video_path': video_path,       
            'question': question,
            'options': options,
            'answer': answer,
            'extra_video_path': extra_video_paths
        }
        return data_item
        
