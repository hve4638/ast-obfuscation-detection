import os
import argparse
import json
import shutil
parser = argparse.ArgumentParser()

parser.add_argument('target')
parser.add_argument('--cache', default=None)
parser.add_argument('-O', '--output', default='./export')

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

def main():
    args = parser.parse_args()
    
    os.makedirs(f'{args.output}/error', exist_ok=True)
    os.makedirs(f'{args.output}/normal', exist_ok=True)

    if args.cache is None:
        result = []
        print(f'walk')
        for dirpath, _, filenames in os.walk(args.target):
            for filename in filenames:
                if filename.endswith('.ps1'):
                    result.append([dirpath, filename])

        count = len(result)
        print(f'find {count} items')
        with open(f'{args.output}/.cache', 'w', encoding='utf-8') as f:
            json.dump(result, f)
        print(f'save cache as {args.output}/.cache')
    else:
        print('read from cache')
        with open(args.cache, 'r', encoding='utf-8') as f:
            result = json.load(f)
        count = len(result)
        print(f'find {count} items')

    try:
        metadata = {}
        maxlen = 0
        for i, (dirpath, filename) in enumerate(result):
            maxlen = max(maxlen, len(filename))
            sublen = maxlen-len(filename)
            print(f'[{i+1}/{count}] ' + filename + ' '*sublen, end='\r')
            
            target = f'{dirpath}\\{filename}'
            outfilename = f'{i}.ps1.txt'
            e = try_open_as_utf8(target)
            if e is None:
                out = f'{args.output}/normal/{outfilename}'
                if not os.path.isfile(out):
                    shutil.copy2(target, out)
                    metadata[outfilename] = target
            elif e is UnicodeDecodeError:
                out = f'{args.output}/error/{outfilename}'
                if not os.path.isfile(out):
                    shutil.copy2(target, out)
                    metadata[outfilename] = target
            else:
                pass
    finally:
        print('Exit')
        with open(f'{args.output}/metadata.json', 'w') as f:
            json.dump(metadata, f)
    return
    for dir, file in datasetpath:
        exportImagePath = f'{args.output}/images/{dir}'
        exportBytesPath = f'{args.output}/bytes/{dir}'
        os.makedirs(exportImagePath, exist_ok=True)
        os.makedirs(exportBytesPath, exist_ok=True)

if __name__ == '__main__':
    main()