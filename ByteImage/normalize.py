import os
import argparse
import json
import shutil
import numpy as np
from PIL import Image
parser = argparse.ArgumentParser()

parser.add_argument('target')
parser.add_argument('-O', '--output', default='./normalize')

def try_open_as_utf8(file_path)->Exception:
    try:
        with open(file_path, encoding='utf-8') as f:
            f.read()
        return None
    except UnicodeDecodeError as e:
        return e
    except Exception as e:
        print(file_path)
        print(f'[Unexpected error] {type(e)} : {e}')
        return e

def export_byte(filename, byte1d):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(byte1d.tolist(), f)

def export_image(filename:str, pixel1d, size=16):
    pixels = [pixel1d[i * size:(i + 1) * size] for i in range(size)]
    
    final_image = Image.fromarray(np.array(pixels, dtype=np.uint8), 'L')
    final_image.save(filename)


def main():
    args = parser.parse_args()
    os.makedirs(f'{args.output}/normalization/', exist_ok=True)
    
    result = []
    for dirpath, _, filenames in os.walk(args.target):
        for filename in filenames:
            if filename.endswith('.json'):
                with open(f'{dirpath}/{filename}', 'r', encoding='utf-8') as f:
                    result.append(json.load(f))

    dataset = np.array(result)
    std = np.std(dataset, axis=0)
    mean = np.mean(dataset, axis=0)
    export_byte(f'{args.output}/normalization/std.json', std)
    export_byte(f'{args.output}/normalization/mean.json', mean)

    std_pixels = np.round(std).astype(np.uint8)
    mean_pixels = np.round(std).astype(np.uint8)
    export_image(f'{args.output}/normalization/std.png', std_pixels, 16)
    export_image(f'{args.output}/normalization/mean.png', mean_pixels, 16)

if __name__ == '__main__':
    main()