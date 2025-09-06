from abc import ABC, abstractmethod

class opt(ABC):
    def __init__(self, **kwargs):
        self.enabled = False
        
    @abstractmethod
    def process(self, opts_name): 
        pass 
        
    def __call__(self, opts_name):
        if self.enabled:
            return self.process(opts_name)
        else:
            return opts_name

    


class combine_type(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('combine_type', None) is not None
        self.combine_type = kwargs.get('combine_type', None)
        self.num_extra_video = kwargs.get('num_extra_video', 0)

    def process(self, opts_name):
        return opts_name + f"_combine_type_{self.combine_type}_num_extra_video_{self.num_extra_video}"

class custom_question(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('custom_question', None) is not None
        self.custom_question = kwargs.get('custom_question', None)

    def process(self, opts_name):
        return opts_name + f"_custom_question_{self.custom_question}"

class add_extra_options(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('add_extra_options', False)

    def process(self, opts_name):
        return opts_name + "_add_extra_options"

class no_target_video(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('no_target_video', False)

    def process(self, opts_name):
        return opts_name + "_no_target_video"

class replace_correct_with_extra(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('replace_correct_with_extra', False)

    def process(self, opts_name):
        return opts_name + "_replace_correct_with_extra"

class shuffle_frame(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('shuffle_frame', False)

    def process(self, opts_name):
        return opts_name + "_shuffle_frame"

class freeze_frame(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('freeze_frame', False)

    def process(self, opts_name):
        return opts_name + "_freeze_frame"

class no_video(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('no_video', False)

    def process(self, opts_name):
        return opts_name + "_no_video"


class shuffle_video(opt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('shuffle_video', False)

    def process(self, opts_name):
        return opts_name + "_shuffle_video"

all_opts = [
    combine_type,
    custom_question,
    add_extra_options,
    no_target_video,
    replace_correct_with_extra,
    shuffle_frame,
    freeze_frame,
    no_video,
    shuffle_video,
]