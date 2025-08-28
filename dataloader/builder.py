

class DataBuilder:
    def __init__(self):
        self._registry = dict()

    def register(self, dataset_name, loader_class):
        self._registry[dataset_name] = loader_class

    def create_loader(self, dataset_name, video_root, data_files=None, shuffle_video=False, num_gpus=1, cur_gpu=0, limit=None, num_extra_video=0, use_local_parquest=False):
        for name, loader_class in self._registry.items():
            if name.lower() in dataset_name.lower():
                return loader_class(video_root, dataset_name, data_files, shuffle_video, num_gpus, cur_gpu, limit, num_extra_video, use_local_parquest)
        raise ValueError(f"Invalid dataset name: {dataset_name}")

data_builder = DataBuilder() 

def register(dataset_name):
    def decorator(cls):
        data_builder.register(dataset_name, cls)
        return cls
    return decorator

