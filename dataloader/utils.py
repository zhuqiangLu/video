import math
try:
    from decord import VideoReader, cpu
except:
    print("decord not installed")

try:
    import av
except:
    print("av not installed")



def split_data(data, num_gpus, limit):
    is_dict = isinstance(data, dict)

    

    if is_dict:
        data = list(data.items())
    elif not isinstance(data, list):
        data = list(data)


    if limit is not None:
        data = data[:int(len(data) * limit)]

    data_size = len(data)
    chunk_size = math.ceil(data_size / num_gpus)  
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    if is_dict:
        chunks = [dict(chunk) for chunk in chunks]

    return chunks



def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def sample_frames(video_path, max_num_frames, start_time=None, end_time=None, backend='decord'):
    # print(backend)
    if backend == 'decord':
        frames = sample_frames_decord(video_path, max_num_frames, start_time, end_time)
        
    elif backend == 'av':
        frames = sample_frames_pyav(video_path, max_num_frames, start_time, end_time)

    return frames 


def sample_frames_decord(video_path, max_num_frames, start_time=None, end_time=None):

    # Load video
    vr = VideoReader(video_path, ctx=cpu(0))

    # Get fps and number of frames
    fps = vr.get_avg_fps()
    num_frames = len(vr)

    # Define time interval (in seconds)
    start_time = start_time if start_time is not None else 0
    end_time   = end_time if end_time is not None else num_frames / fps

    # Convert time interval into frame indices
    start_idx = int(start_time * fps)
    end_idx   = int(end_time * fps)

    # Sample 1 frame per second in the interval
    frame_idx = [int(i * fps) for i in range(int(start_time), int(end_time))]

    # Make sure indices are within video range
    frame_idx = [idx for idx in frame_idx if idx < num_frames]

    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


def sample_frames_pyav(video_path, max_num_frames, start_time=None, end_time=None):
    """
    Sample frames between start_time and end_time at a given fps.
    
    Args:
        video_path (str): Path to video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        fps (float): Desired sampling fps.
        
    Returns:
        List of (timestamp, frame_ndarray).
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Seek close to the start time (in pts units)
    if start_time is not None:
        container.seek(int(start_time / video_stream.time_base))
        next_sample_time = start_time
    else:
        next_sample_time = 0 


    frames = []

    # sample_fps = round(container.streams.video[0].average_rate / 1)  # FPS
    # step = 1.0 / fps  # interval between samples (seconds)
    step = 1.0 # sample at 1 fps

    for frame in container.decode(video_stream):
        timestamp = frame.pts * video_stream.time_base

        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            break


        if timestamp >= next_sample_time:
            img = frame.to_image()
            frames.append(img)
            next_sample_time += step

    duration = float(container.duration/ av.time_base)
    container.close()
    # print(f'sample {len(frames)} frames from video with duration {duration:.2f}s from {video_path}, start_time {start_time}, end_time {end_time}')
    if len(frames) > max_num_frames:
        # Uniformly sample frames to reduce to max_num_frames
        indices = list(range(len(frames)))
        sample_indices = uniform_sample(indices, max_num_frames)
        frames = [frames[i] for i in sample_indices]
    return frames  



  
