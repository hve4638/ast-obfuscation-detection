from collections import Counter
import os
import numpy as np
import argparse
import utils
parser = argparse.ArgumentParser()

parser.add_argument('filename')
parser.add_argument('--cache', default=None)
parser.add_argument('-O', '--output', default='./export')

args = parser.parse_args()

def get_export_path(export_path:str, *paths:list[str])->str:
    filename = utils.remove_parentdirs(os.path.join(*paths))\
        .replace('./', '').replace('.\\', '')\
        .replace('\\', '#').replace('/', '#')
    return os.path.join(export_path, filename)

def extract_byte(filename):
    try:
        utils.assert_utf8(filename)
        with open(filename, 'rb') as f:
            data = f.read()
        return list(data)
    except Exception as e:
        print('error', e)
        return None

def main():
    if args.cache is not None and os.path.isfile(args.cache):
        cache = utils.read_json(args.cache)
        ls = cache['ls']
    else:
        ls = utils.get_path_recursive(args.filename, lambda x: x.endswith('.py'))
        cache = {
            'filename' : args.filename,
            'ls' : ls
        }
        if args.cache is not None:
            utils.write_json(args.cache, cache)
    
    os.makedirs(args.output, exist_ok=True)
    for dirpath, filename in ls:
        target_path = os.path.join(dirpath, filename)
        byte_sequence = extract_byte(target_path)
        if byte_sequence is not None:
            output_path = get_export_path(args.output, dirpath, filename + '.bytes')
            utils.write_json(output_path, byte_sequence)


if __name__ == '__main__':
    main()
    