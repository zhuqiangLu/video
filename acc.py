import os 
import json
import sys
from glob import glob 
import numpy as np 


if __name__ == '__main__':
    src_dir = sys.argv[1]
    all_jonsl = glob(os.path.join(src_dir, '*.jsonl')) 
    acc = list()
    for jonsl in all_jonsl:
        with open(jonsl, 'r') as f:
            for line in f:
                item = json.loads(line)
                acc.append(item['acc']) 
    print(np.mean(acc))    
