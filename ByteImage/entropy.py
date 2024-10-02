from collections import Counter
import os
import numpy as np
from PIL import Image
import sys
import math
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('filename')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--log', action='store_true')
parser.add_argument('-O', '--output', default='./export')

args = parser.parse_args()

def get_dataset_path(dirpath):
    result = []
    for dirpath, _, filenames in os.walk(dirpath):
        result.extend((dirpath, filename) for filename in filenames)
    return result

def count_byte_frequency(filename:str) -> Counter:
    with open(filename, 'rb') as f:
        data = f.read()
    byte_groups = [byte for byte in data]
    frequency = Counter(byte_groups)
    return frequency

def convert_linear(value, max_value):
    return round(value * 255 / max_value)

def convert_log(value, base):
    try:
        if base == 1.0:
            return 255 if value > 0 else 0
        elif value >= 1:
            return round(
                math.log(value, base)*127.5
            )
        else:
            return 0
        return round(
                math.log(value, base)*127.5 if value > 1 else 0
            )
    except ZeroDivisionError:
        print('Error #ZeroDivisionError')
        print(value, base)
        raise

def export_image(filename:str, byte_counter:Counter):
    size = 16
    max_prob = byte_counter.most_common(1)[0][1]
    base = math.sqrt(max_prob)
    if args.linear:
        pixel1d = [convert_linear(byte_counter[i], max_prob) for i in range(256)]
    else:
        pixel1d = [convert_log(byte_counter[i], base) for i in range(256)]
    
    pixels = [pixel1d[i * size:(i + 1) * size] for i in range(size)]
    if args.verbose:
        for pixel1d in pixels:
            print(pixel1d)
    final_image = Image.fromarray(np.array(pixels, dtype=np.uint8), 'L')
    final_image.save(filename)

def print_byte_frequency(byte_counter:Counter):
    max_prob = byte_counter.most_common(1)[0][1]
    for value, frequency in byte_counter.most_common():
        print("0x{:02x}: {:<6} {}".format(value, frequency, "█" * int(frequency * 80/max_prob)))

datasetpath = get_dataset_path(args.filename)

for dir, file in datasetpath:
    os.makedirs(f'{args.output}/{dir}', exist_ok=True)
    counter = count_byte_frequency(f'{dir}/{file}')
    if args.verbose:
        print(f'{file}')
        print_byte_frequency(counter)
        print()
    try:
        export_image(f'./{args.output}/{dir}/{file}.png', counter)
    except:
        print('Error Occured!')
        print(f'{dir}/{file}')
        break