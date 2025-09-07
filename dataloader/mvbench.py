import os 
from datasets import load_dataset
from torch.utils.data import Dataset
import random
from .utils import split_data
from .builder import register
import json
import numpy as np 
import torchvision.transforms as T
from .mvbench_utils import (
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    GroupNormalize,
)
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from PIL import Image
import imageio
import cv2


data_list = {
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "pMoments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        # torch_imgs = self.transform(images_group)
        return images_group
    
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        # torch_imgs = self.transform(images_group)
        return images_group
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        # torch_imgs = self.transform(images_group)
        return images_group

    def qa_template(self, data):
        # question = f"Question: {data['question']}\n"
        # question += "Options:\n"
        question = data['question']
        answer = data['answer']
        answer_idx = -1
        options = list()
        for idx, c in enumerate(data['candidates']):
            options.append(f"{chr(ord('A') + idx)}.{c}")
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, options, answer

    

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        pil_imgs = decord_method(video_path, bound)
        question, options, answer = self.qa_template(self.data_list[idx]['data'])

            
        return {
            'video': pil_imgs, 
            'question': question, 
            'options': options,
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

    




@register("MVBench")
class MVBenchDataset(Dataset):
    def __init__(self, video_root, dataset_name='lmms-lab/MVBench', data_files=None, shuffle_video=False, num_gpus=1, cur_gpu=0, limit=None, num_extra_video=0, use_local_parquest=False, backend='decord', max_num_frames=10):
        # print(f"Loading dataset from {data_files}")
        assert use_local_parquest, 'mvbench only support local files'
        
        new_data_list = dict()
        for key, value in data_list.items():
            new_value = [item for item in value] 
            new_value[1] = os.path.join(video_root, new_value[1])
            new_data_list[key] = new_value

        
        self.dataset = MVBench_dataset(data_files, new_data_list, num_segments=max_num_frames)
        
        
        self.shuffle_video = shuffle_video

        indices = list(range(len(self.dataset)))
        self._indices = split_data(indices, num_gpus, limit)[cur_gpu]
        self.num_extra_video = num_extra_video
        
        


    def __len__(self):
        return len(self._indices)

    
    def __getitem__(self, idx):

        

        _idx = self._indices[idx]
        if self.shuffle_video:
            print('shuffle video')
            _idx = random.choice(self._indices)

        item = self.dataset[_idx]

        frames = item['video']
        
        

        extra_video_idxs = random.sample(self._indices, self.num_extra_video)

        extra_frames = [self.dataset[extra_video_idx]['video'] for extra_video_idx in extra_video_idxs]

       


        data_item = {
            'video_path': None,   
            'frames': frames,
            'extra_frames': extra_frames,
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer'],
            'extra_video_path': None
        }
        return data_item
    
        