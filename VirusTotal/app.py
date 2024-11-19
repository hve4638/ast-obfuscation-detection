import os
import csv, json
import math
import re
from upload import upload_vt
from secret import api_key

dataset_paths = [
    '../../unpack/attack_base64decoded',
    '../../unpack/attack_original',
]
output_path = '../../vt_result/'

pass_count = 0
error_count = 0
skip_count = 0
for dataset_path in dataset_paths:
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            target = os.path.join(dirpath, filename)
            _, lastpath = os.path.split(dirpath)
            outputdir = os.path.join(output_path, lastpath)
            output = os.path.join(outputdir, filename + '.json')
            os.makedirs(outputdir, exist_ok=True)
            
            if not os.path.isfile(output):
                report = upload_vt(target, api_key)
                if report is None:
                    error_count += 1
                    print(f'error : {target}')
                else:
                    pass_count += 1
                    print(f'VT : {target}')
                    with open(output, 'w') as f:
                        json.dump(report, f, indent=4)
            else:
                skip_count += 1
                
print(f'pass: {pass_count}')
print(f'skip: {skip_count}')
print(f'error: {error_count}')