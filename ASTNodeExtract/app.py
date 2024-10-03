import subprocess
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('target')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('-O', '--output', default='../export/ASTNode')

args = parser.parse_args()

def extract_astnode(filename):
    ps1_script_path = os.path.abspath('./astextract.ps1')
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1_script_path, filename],
        capture_output=True, text=True
    )

    return result.stdout

if __name__ == '__main__':

    filepaths = []
    for dirpath, _, filenames in os.walk(args.target):
        for filename in filenames:
            filepaths.append((dirpath, filename))
    
    for dirpath, filename in filepaths:
        path = os.path.join(dirpath, filename)
        exportpath = os.path.join(args.output, dirpath, filename)
        os.makedirs(os.path.join(args.output, dirpath), exist_ok=True)
        nodes = extract_astnode(path)
        with open(f'{exportpath}.txt', 'w') as f:
            f.write(nodes)
        