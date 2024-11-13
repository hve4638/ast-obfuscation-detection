import os
import csv, json
import math
import re
from upload import upload_vt

dataset_paths = [
    '../../unpack/attack_base64decoded',
    '../../unpack/attack_original',
]
output_path = '../../vt_result/'

for dataset_path in dataset_paths:
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            target = os.join(dirpath, filename)
            output = os.join(output_path, dirpath, filename + '.json')
            print(target)