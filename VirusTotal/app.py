import os, sys
import csv, json
import math
import re, time
from upload import upload_vt
from secret import api_key

PASS = 0
FAIL = 1
SKIP = 2

# 공격 데이터셋 경로
input_paths = [
    './dataset',
]
# 출력 경로
output_path = './export'

# 데이터셋 범위
DATASET_RANGE = (2000, 3000)

re_filename = re.compile(r'^#([0-9]+)_.*$')
def filename_filter(filename):
    if m := re_filename.match(filename):
        no = int(m[1])
        if no >= DATASET_RANGE[0] and no <= DATASET_RANGE[1]:
            return True
        else:
            return False
    else:
        return False

def proc(target, output):
    if os.path.isfile(output):
        return SKIP
    
    try:
        report = upload_vt(target, api_key)
    except:
        sys.stderr.write('API rate limit exceeded. Wait 1 hour.\n')
        time.sleep(60*60)
        return FAIL

    if report is None:
        return FAIL
    else:
        with open(output, 'w') as f:
            json.dump(report, f, indent=4)
        return PASS

def main():
    pass_count = 0
    fail_count = 0
    skip_count = 0
    for input_path in input_paths:
        for dirpath, _, filenames in os.walk(input_path):
            for filename in filenames:
                if not filename_filter(filename):
                    continue
                target = os.path.join(dirpath, filename)
                _, lastpath = os.path.split(dirpath)
                outputdir = os.path.join(output_path, lastpath)
                output = os.path.join(outputdir, filename + '.json')
                os.makedirs(outputdir, exist_ok=True)
                
                status = proc(target, output)
                if status == PASS:
                    pass_count += 1
                    sys.stdout.write(f'pass : {target}\n')
                elif status == FAIL:
                    fail_count += 1
                    sys.stdout.write(f'fail : {target}\n')
                elif status == SKIP:
                    skip_count += 1
                    
    return pass_count, fail_count, skip_count

if __name__ == '__main__':
    while True:
        pass_count, fail_count, skip_count = main()
        print(f'pass: {pass_count}')
        print(f'fail: {fail_count}')
        print(f'skip: {skip_count}')

        if pass_count == 0 and fail_count == 0:
            print('')
            print('모든 데이터셋 처리됨')
            print('이외 데이터셋을 처리하려면 DATASET_RANGE를 수정하세요')
            break