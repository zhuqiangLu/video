import os 
from datasets import load_dataset
from torch.utils.data import Dataset
import random
from .utils import split_data



class VideoEvalProDataset(Dataset):
    def __init__(self, dataset_path, dataset_name='TIGER-Lab/VideoEval-Pro', data_files=None, shuffle_video=False, num_gpus=1, cur_gpu=0, limit=None, num_extra_video=0):
        print(dataset_name, data_files)
        self.dataset = load_dataset(dataset_name, split='test', data_files={'test': data_files})
        # self.dataset = load_dataset(dataset_name, split='test')
        # self.dataset = load_dataset("parquet", data_files=data_files)

        self.video_root = os.path.join(dataset_path)
        self.shuffle_video = shuffle_video
        self.video_list = os.listdir(self.video_root)
        self.num_extra_video = num_extra_video
        self.dataset = split_data(self.dataset, num_gpus, limit)[cur_gpu]
        self.num_extra_video = num_extra_video
        
        # if limit is not None:
        #     self.dataset = self.dataset[:int(limit*len(self.dataset))]
            # print(len(self.dataset))



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.shuffle_video:
            video_path = random.choice(os.listdir(self.video_root))
            video_path = os.path.join(self.video_root, video_path)
        else:
            video_path = os.path.join(self.video_root, item['video'])

        extra_video_paths = [os.path.join(self.video_root, p) for p in random.sample(os.listdir(self.video_root), self.num_extra_video)]

        question = item['question']
        options = item['options']
        answer = item['answer']
        

        

        data_item = {
            'video_path': video_path,       
            'question': question,
            'options': options,
            'answer': answer,
            'extra_video_path': extra_video_paths
        }
        return data_item
        
