from collections import Counter
import os, csv
import numpy as np
import argparse
import utils
parser = argparse.ArgumentParser()

parser.add_argument('filename')
parser.add_argument('--cache', default=None)
parser.add_argument('-O', '--output', default='./export')
parser.add_argument('--csv', default=None)
parser.add_argument('--ext', default='.ps1')

args = parser.parse_args()

def get_export_path(export_path:str, *paths:list[str])->str:
    filename = utils.remove_parentdirs(os.path.join(*paths))\
        .replace('./', '').replace('.\\', '')\
        .replace('\\', '#').replace('/', '#')
    return os.path.join(export_path, filename)

def extract_byte(filename):
    try:
        data = utils.read_bytes_as_utf8(filename)
        return list(data)
    except Exception as e:
        print(filename, e)
        return None

def main():
    if args.cache is not None and os.path.isfile(args.cache):
        cache = utils.read_json(args.cache)
        ls = cache['ls']
    else:
        ls = utils.get_path_recursive(args.filename, lambda x: x.endswith(args.ext))
        cache = {
            'filename' : args.filename,
            'ls' : ls
        }
        if args.cache is not None:
            utils.write_json(args.cache, cache)

    if not ls:
        abspath = os.path.abspath(args.filename)
        print(f"No files found : '{abspath}'")
        if args.cache:
            os.remove(args.cache)
    
    if args.csv is not None:
        write_as_csv(args.csv, ls)
    else:
        write_as_splite_file(args.output, ls)

def write_as_csv(export_path, ls):
    export_directory, _ = os.path.split(export_path)
    
    os.makedirs(export_directory, exist_ok=True)
    with open(export_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'filename', 'bytes'])
        
        for dirpath, filename in ls:
            target_path = os.path.join(dirpath, filename)
            byte_sequence = extract_byte(target_path)
            if byte_sequence is not None:
                writer.writerow([dirpath, filename, byte_sequence])

def write_as_splite_file(export_directory, ls):
    os.makedirs(export_directory, exist_ok=True)
    for dirpath, filename in ls:
        target_path = os.path.join(dirpath, filename)
        byte_sequence = extract_byte(target_path)
        if byte_sequence is not None:
            output_path = get_export_path(export_directory, dirpath, filename + '.bytes')
            utils.write_json(output_path, byte_sequence)

if __name__ == '__main__':
    main()
    