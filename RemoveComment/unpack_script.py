import argparse
import csv, json
import os, sys

csv.field_size_limit(10000000)

parser = argparse.ArgumentParser()

parser.add_argument('filename')
parser.add_argument('--output', '-O', default='./export')
parser.add_argument('--txt', action='store_true')

args = parser.parse_args()

_, target = os.path.split(args.filename)
target, _ = os.path.splitext(target)

output_dir = os.path.join(args.output, target)
os.makedirs(output_dir, exist_ok=True)

with open(args.filename, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        b = bytes(json.loads(row['bytes']))
        
        output_path = os.path.join(output_dir, f'#{i}_{row['filename']}')
        if args.txt:
            output_path += '.txt'
        with open(output_path, 'wb') as f:
            f.write(b)
