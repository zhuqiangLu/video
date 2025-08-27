import math
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