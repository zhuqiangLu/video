

class DataBuilder:
    def __init__(self):
        self._registry = dict()

    def register(self, dataset_name, loader_class):
        self._registry[dataset_name] = loader_class

    def create_loader(self, **kwargs):
        dataset_name = kwargs.get('dataset_name', None)
        video_root = kwargs.get('video_root', None)
        data_files = kwargs.get('data_files', None)
        shuffle_video = kwargs.get('shuffle_video', False)
        num_gpus = kwargs.get('num_gpus', 1)
        cur_gpu = kwargs.get('cur_gpu', 0)
        limit = kwargs.get('limit', None)
        num_extra_video = kwargs.get('num_extra_video', 0)
        use_local_parquest = kwargs.get('use_local_parquest', False)
        backend = kwargs.get('backend', 'decord')
        max_num_frames = kwargs.get('max_num_frames', 10)

        assert dataset_name is not None, "dataset_name is required" 
        assert video_root is not None, "video_root is required" 

        for name, loader_class in self._registry.items():
            if name.lower() in dataset_name.lower():
                return loader_class(video_root, dataset_name, data_files, shuffle_video, num_gpus, cur_gpu, limit, num_extra_video, use_local_parquest, backend, max_num_frames)
        raise ValueError(f"Invalid dataset name: {dataset_name}")

data_builder = DataBuilder() 

def register(dataset_name):
    def decorator(cls):
        data_builder.register(dataset_name, cls)
        return cls
    return decorator

