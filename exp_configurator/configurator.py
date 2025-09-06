import os 
import json
from .exp import all_exp


class configurator():
    def __init__(self, **kwargs):
        self.config = kwargs

        self.max_num_frames = kwargs.get('max_num_frames', 10)
        self.backend = kwargs.get('backend', 'decord')
        self.result_dir = kwargs.get('result_dir', 'results')
        self.model_base = kwargs.get('model_base', None)
        self.cur_gpu = kwargs.get('cur_gpu', 0) 
        self.dataset_name = kwargs.get('dataset_name', None) 
        self.exp = [exp(**kwargs) for exp in all_exp]
        self._log_root = self._get_log_root()
        os.makedirs(self._log_root, exist_ok=True)
        self.save_config() 


        
        



    def save_config(self):
        if self.cur_gpu is 0:
            with open(f"{self._log_root}/config.json", "w") as f:
                json.dump(self.config, f, indent=4)

        
    def get_log_path(self):
        return f"{self._log_root}/{self.cur_gpu}.jsonl"


    def configure_inputs(self, item):
        frames = item['frames']
        extra_frames = item['extra_frames']
        question = item["question"]
        options = item["options"]
        answer = item["answer"]

        for exp in self.exp:
            question, options, answer, frames, extra_frames = exp(question, options, answer, frames, extra_frames)

        return question, options, answer, frames, extra_frames

    def _get_log_root(self):

        '''
       Manage the saved file name 
        '''
        _log_root = f"{self.result_dir}/{self.model_base.replace('/', '-')}"
        for exp in self.exp:
            _log_root = exp.add_opts(_log_root)

        return _log_root


        