from collections import Counter
import os
import numpy as np
from PIL import Image
import sys
import math
import argparse
import json
parser = argparse.ArgumentParser()

parser.add_argument('filename')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('-O', '--output', default='./export')

args = parser.parse_args()

def get_dataset_path(dirpath):
    '''
    디렉토리
    '''
    result = []
    for dirpath, _, filenames in os.walk(dirpath):
        result.extend((dirpath, filename) for filename in filenames)
    return result

def count_byte_frequency(filename:str) -> Counter:
    '''
    파워쉘 스크립트 파일을 1바이트 단위로 읽고 각 바이트의 빈도수 계산
    바이트는 0x00~0xFF(0~255) 범위 사이이므로 총 256개의 바이트 종류에 대한 빈도수를 계산
    '''
    with open(filename, 'rb') as f:
        data = f.read()
    byte_groups = [byte for byte in data]
    frequency = Counter(byte_groups)
    return frequency

def convert_linear(value:int, max_value:int):
    '''
    선형 변환
    '''
    return round(value * 255 / max_value)

def convert_log(value:int, base:int):
    '''
    로그 변환
    '''
    try:
        if base == 1.0:
            # log의 밑이 1일 수 없으므로 분기 처리
            # 스크립트가 짧은 경우 base가 1 인 케이스가 자주 발생함
            return 255 if value > 0 else 0
        elif value >= 1:
            # 0~255 사이의 값으로 변환됨
            return round(math.log(value, base)*127.5)
        else:
            return 0
    except ZeroDivisionError:
        print('Error #ZeroDivisionError')
        print(value, base)
        raise

def parse_byte_frequency(byte_counter:Counter):
    '''
    각 바이트 빈도수를 기준으로 bytes 이미지를 생성하기 위한 1차원 배열 생성후 반환
    '''
    # 16*16 이미지 기준이므로 size를 16으로 지정
    size = 16
    
    # 최대 빈도수
    max_prob = byte_counter.most_common(1)[0][1]

    # 그레이스케일 이미지로 변환하기 위해 빈도수 배열 데이터를 0-255 사이의 값으로 변환
    # 최대 빈도수를 가진 값을 255로 만드는 변환 방법을 사용
    if args.linear:
        # 선형 변환
        pixel1d = [convert_linear(byte_counter[i], max_prob) for i in range(256)]
    else:
        # 로그 변환
        base = math.sqrt(max_prob)
        pixel1d = [convert_log(byte_counter[i], base) for i in range(256)]
    
    pixels = [pixel1d[i * size:(i + 1) * size] for i in range(size)]
    if args.verbose:
        for pixel1d in pixels:
            print(pixel1d)
    return pixels

def export_image(filename:str, pixel1d, size=16):
    pixels = [pixel1d[i * size:(i + 1) * size] for i in range(size)]
    
    final_image = Image.fromarray(np.array(pixels, dtype=np.uint8), 'L')
    final_image.save(filename)

def export_json(filename:str, pixel1d):
    with open(filename, 'w') as f:
        json.dump(pixel1d, f)

def print_byte_frequency(byte_counter:Counter):
    '''
    디버깅용
    '''
    max_prob = byte_counter.most_common(1)[0][1]
    for value, frequency in byte_counter.most_common():
        print("0x{:02x}: {:<6} {}".format(value, frequency, "█" * int(frequency * 80/max_prob)))

def remove_parentdirs(target_path:str):
    return target_path.replace('../').replace('..\\')

if __name__ == '__main__':
    datasetpath = get_dataset_path(args.filename)

    for dir, file in datasetpath:
        exportImagePath = f'{args.output}/images/{dir}'
        exportBytesPath = f'{args.output}/bytes/{dir}'
        os.makedirs(exportImagePath, exist_ok=True)
        os.makedirs(exportBytesPath, exist_ok=True)
        outdir = remove_parentdirs(dir)

        # 바이트 빈도수 계산
        counter = count_byte_frequency(f'{dir}/{file}')

        if args.verbose:
            print(f'{file}')
            print_byte_frequency(counter)
        
        try:
            pixel1d = parse_byte_frequency(counter)
            
            export_image(f'./{exportImagePath}/{file}.png', pixel1d)
            export_json(f'./{exportBytesPath}/{file}.png', pixel1d)
        except:
            print('Error Occured!')
            print(f'{dir}/{file}')
            break