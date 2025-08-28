import os 
import json
from torch.utils.data import Dataset
import random
from .utils import split_data
from pprint import pprint
from .builder import register


@register("MotionBench")
class MotionBenchDataset(Dataset):
    def __init__(self, dataset_path, dataset_name=None, data_files=None, shuffle_video=False, num_gpus=1, cur_gpu=0, limit=None, num_extra_video=0, use_local_parquest=False, **kwargs):

        print('WARNING: MotionBenchDataset will ignore argment: dataset_name, but instead uses the video_info.meta.jsonl')
        # self.dataset = load_dataset(dataset_name, split='test')
        # self.dataset = load_dataset('jsonl', data_files=os.path.join(dataset_path, 'video_info.meta.jsonl'))
        with open(os.path.join(dataset_path, 'video_info.meta.jsonl'), 'r') as f:
            raw_dataset = [json.loads(line) for line in f]

        self.video_root = os.path.join(dataset_path)
        self.dataset = self.preprocess(raw_dataset)

        self.shuffle_video = shuffle_video
        self.video_list = os.listdir(self.video_root)
        self.num_extra_video = num_extra_video
        self.dataset = split_data(self.dataset, num_gpus, limit)[cur_gpu]
        self.num_extra_video = num_extra_video

        self.all_video_paths = [item['video_path'] for item in self.dataset]
        
        # if limit is not None:
        #     self.dataset = self.dataset[:int(limit*len(self.dataset))]
            # print(len(self.dataset))

    def preprocess(self, raw_dataset):
        dataset = []
        # print(len(raw_dataset))
        total_count = 0
        valid_count = 0
        for item in raw_dataset:
            qa = item['qa'] 

            for qa_item in qa:
                
                answer = qa_item['answer']
                is_path_valid = False 
                valid_path = None
                self_collected_path = os.path.join(self.video_root, 'self-collected', item['video_path'])
                public_dataset_path = os.path.join(self.video_root, 'public-dataset', item['video_path'])
                if os.path.exists(self_collected_path):
                    is_path_valid = True
                    valid_path = self_collected_path
                elif os.path.exists(public_dataset_path):
                    is_path_valid = True
                    valid_path = public_dataset_path
                else:
                    is_path_valid = False 

                total_count += 1
                
                if answer != 'NA' and is_path_valid:
                    valid_count += 1 
                    start = qa_item['start']
                    end = qa_item['end']
                    question_with_opt = qa_item['question'].split('\n')
                    question = question_with_opt[0]
                    options = question_with_opt[1:] 
                

                    single_item = dict() 
                    single_item['video_path'] = valid_path
                    single_item['video_type'] = item['video_type']
                    single_item['question_type'] = item['question_type']
                    single_item['question'] = question
                    single_item['options'] = options
                    single_item['answer'] = answer
                    single_item['start'] = start
                    single_item['end'] = end
                    single_item['video_info'] = item['video_info']
                    
                    
                    dataset.append(single_item)

        print(f"Total count: {total_count}, Valid DEV count: {valid_count}")
        return dataset



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.shuffle_video:
            print('shuffle video')
            video_path = random.choice(self.all_video_paths)
            video_path = os.path.join(self.video_root, video_path)
        else:
            video_path = os.path.join(self.video_root, item['video_path'])

        extra_video_paths = [os.path.join(self.video_root, p) for p in random.sample(os.listdir(self.video_root), self.num_extra_video)]
        
        # for qa_item in qa:
            
        question = item['question']
        options = item['options']
        answer = item['answer']
        start = item['start']
        end = item['end']

        
        
        
        

        

        data_item = {
            'video_path': video_path,       
            'question': question,
            'options': options,
            'answer': answer,
            'extra_video_path': extra_video_paths,
            "start_end": (start, end),
            "extra_info": {"question_type": item['question_type'], "video_type": item['video_type'], "video_info": item['video_info']}
        }
        return data_item
        
