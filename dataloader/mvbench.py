import os 
from datasets import load_dataset
from torch.utils.data import Dataset
import random
from .utils import split_data
from .builder import register


@register("MVBench")
class MVBenchDataset(Dataset):
    def __init__(self, video_root, dataset_name='lmms-lab/MVBench', data_files=None, shuffle_video=False, num_gpus=1, cur_gpu=0, limit=None, num_extra_video=0, use_local_parquest=False):
        # print(f"Loading dataset from {data_files}")
        assert use_local_parquest, 'mvbench only support local files'
        self.dataset = dict()
        for file in os.listdir(data_files):
            self.dataset[file.replace('.json', '')] = load_dataset("json", split='test', data_files={'test': os.path.join(data_files, file)})
            # self.dataset = load_dataset("json", split='test', data_files={'test': os.path.join(data_files, file)})
    
        print(self.dataset)
        raise
        
        self.video_root = os.path.join(video_root)
        self.shuffle_video = shuffle_video
        self.video_list = os.listdir(self.video_root)
        self.dataset = split_data(self.dataset, num_gpus, limit)[cur_gpu]
        self.num_extra_video = num_extra_video
        
        


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        

        item = self.dataset[idx]
        if self.shuffle_video:
            print('shuffle video')
            video_path = random.choice(os.listdir(self.video_root))
            video_path = os.path.join(self.video_root, video_path)
        else:
            video_path = os.path.join(self.video_root, item['videoID']+".mp4")

        extra_video_paths = [os.path.join(self.video_root, p) for p in random.sample(os.listdir(self.video_root), self.num_extra_video)]


        item = self.dataset[idx]
        data_item = {
            'video_path': video_path,       
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer'],
            'extra_video_path': extra_video_paths
        }
        return data_item
    
        