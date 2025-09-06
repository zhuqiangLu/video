from abc import ABC, abstractmethod
import random 
import numpy as np

class exp(ABC):
    def __init__(self, **kwargs):
        self.enabled = False
        self.opts = ""
        
    @abstractmethod
    def process(self, question, options, answer, frames, extra_frames): 
        pass 


    def add_opts(self, opts):
        return opts + self.opts
        
    def __call__(self, question, options, answer, frames, extra_frames):
        if self.enabled:
            return self.process(question, options, answer, frames, extra_frames)
        else:
            return question, options, answer, frames, extra_frames

    


class video_position(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('custom_question', "") == 'video_position'
        self.combine_type = kwargs.get('combine_type', None)

        self.opts = f"|custom_question:video_position" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        option_num = ord(answer) - ord('A') 

        question = f"Which part of the video is most relevant to the given term: {options[option_num]}?\n"
        options = ["A.beginning", "B.middle", "C.end"]
        if self.combine_type == 'target_first':
            answer = "A"
        elif self.combine_type == 'target_last':
            answer = "C"
        elif self.combine_type == 'target_middle':
            answer = "B"
        else:
            raise ValueError(f"Invalid combine_type: {self.combine_type}")
        return question, options, answer, frames, extra_frames



class video_number(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('custom_question', "") == 'video_number'
        self.opts = f"|custom_question:video_number" if self.enabled else ""


    def process(self, question, options, answer, frames, extra_frames):
        question = f"The given video is combined by multiple videos. How many videos are combined?\n"
        correct_num = len(extra_frames)+1 
        # Generate 3 random numbers from 0-10 excluding correct_num
        possible_nums = list(range(11))
        possible_nums.remove(correct_num)
        wrong_nums = np.random.choice(possible_nums, size=3, replace=False)
        options = [f"A.{wrong_nums[0]}", f"B.{wrong_nums[1]}", f"C.{wrong_nums[2]}", f"D.{correct_num}"]
        answer = "D" 
        return question, options, answer, frames, extra_frames


class count_frame(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('custom_question', "") == 'count_frame'
        self.max_num_frames = kwargs.get('max_num_frames', None)
        assert self.max_num_frames is not None, "max_num_frames is required"
        self.opts = f"|custom_question:count_frame" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        question = f"How many frames are in the video?\n"
        possible_nums = list(range(64))
        possible_nums.remove(self.max_num_frames)   
        wrong_nums = np.random.choice(possible_nums, size=3, replace=False)
        options = [f"A.{wrong_nums[0]}", f"B.{wrong_nums[1]}", f"C.{wrong_nums[2]}", f"D.{self.max_num_frames}"]
        answer = "D" 
        return question, options, answer, frames, extra_frames


class frozen_vieo_bool(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('custom_question', "") == 'frozen_video_bool'
        self.frozen_video = kwargs.get('frozen_video', False)
        self.opts = f"|frozen_video_bool" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        question = f"Is the given video frozen? \n"
        options = ["A.True", "B.False"]
        if self.frozen_video:
            answer = "A"
        else:
            answer = "B"
        return question, options, answer, frames, extra_frames

class replace_correct_with_extra(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('replace_correct_with_extra', False)
        self.opts = f"|replace_correct_with_extra" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        option_num = ord(answer) - ord('A') 
        opt_let = answer
        options[option_num] =  f'{opt_let}.None of the above'
        options = options
        return question, options, answer, frames, extra_frames


class add_extra_options(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get(add_extra_options, False) 
        assert not (kwargs.get('custom_question', False) and kwargs.get('replace_correct_with_extra', False)), "add_extra_options and replace_correct_with_extra cannot be True at the same time"
        self.no_target_video = kwargs.get('no_target_video', False)
        self.opts = f"|add_extra_options" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        option_letter = chr(64 + len(options)) if len(options) <= 2 else chr(96 + len(options))
        options = options + [f'{option_letter.upper()}. None of the above']
        # if no target video, that means all answer are wrong 
        if self.no_target_video:
            answer = option_letter.upper()
        return question, options, answer, frames, extra_frames

'''
Frame related

'''
class combine_frame(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('combine_type', None) is not None
        self.combine_type = kwargs.get('combine_type', None)
        self.opts = f"|combine_frame:{self.combine_type}" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        assert type(extra_frames) == list, "extra_frames must be a list"
        assert type(frames) == list, "frames must be a list"

        if self.combine_type == 'target_first' or self.combine_type == None:
            extra_frames.insert(0, frames)
        elif self.combine_type == 'target_last':
            extra_frames.insert(len(extra_frames)+1, frames)
        elif self.combine_type == 'target_middle':
            assert len(extra_frames) >= 2, "extra_frames must be 1 when combine_type is target_middle"
            extra_frames.insert(len(extra_frames)//2, frames)
        else:
            raise ValueError(f"Invalid combine_type: {self.combine_type}")

        all_frames =list() 
        for item in extra_frames:
            all_frames.extend(item) 

        # finally we unify the frame resolution 
        if len(all_frames) > 0:
            frame_res = frames[0].size
            for idx, frame in enumerate(all_frames):
                all_frames[idx] = frame.resize(frame_res)

        return question, options, answer, all_frames, []

class shuffle_frame(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('shuffle_frame', False)
        self.opts = f"|shuffle_frame" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        assert type(frames) == list, "frames must be a list"
        assert type(extra_frames) == list, "extra_frames must be a list"

        if self.shuffle_frame:
            random.shuffle(frames)
        for item in extra_frames:
            random.shuffle(item)

        return question, options, answer, frames, extra_frames

class freeze_frame(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('freeze_frame', False)
        self.opts = f"|freeze_frame" if self.enabled else ""

    def process(self, question, options, answer, frames, extra_frames):
        assert type(frames) == list, "frames must be a list"
        assert type(extra_frames) == list, "extra_frames must be a list"
        random_frame = random.choice(frames)
        frozen_frames = [random_frame] * len(frames)

        frozen_extra_frames = list() 
        for item in extra_frames:
            random_frame = random.choice(item)
            frozen_extra_frames.append([random_frame] * len(item))

        return question, options, answer, frozen_frames, frozen_extra_frames


class no_target_video(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('no_target_video', False)
        self.opts = f"|no_target_video" if self.enabled else ""
        
    def process(self, question, options, answer, frames, extra_frames):
        return question, options, answer, [], extra_frames


class no_video(exp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = kwargs.get('no_video', False)
        self.opts = f"|no_video" if self.enabled else ""
       
    def process(self, question, options, answer, frames, extra_frames):
        return question, options, answer, [], []




all_exp = [
    video_position,
    video_number,
    count_frame,
    frozen_vieo_bool,
    replace_correct_with_extra,
    add_extra_options,
    shuffle_frame,
    freeze_frame,
    no_video,
    combine_frame, #combine frame must be the last one
]