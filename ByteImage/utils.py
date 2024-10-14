import os
import json

def read_json(filename:str) -> dict|list:
    with open(filename, 'r') as f:
        return json.load(f)

def write_json(filename:str, jsondata:dict|list):
    with open(filename, 'w') as f:
        json.dump(jsondata, f)

def read_text(filename:str) -> str:
    with open(filename, 'r') as f:
        return f.read()
    
def write_text(filename:str, text:str):
    with open(filename, 'w') as f:
        f.write(text)
        
def remove_parentdirs(target_path:str)->str:
    '''
    Remove parent directories from the path.
    '''
    return target_path.replace('../', './').replace('..\\', '.\\')

def get_path_recursive(target, on_filter=None)->list[tuple[str, str]]:
    on_filter = on_filter or (lambda x: True)

    result = []
    for dirpath, _, filenames in os.walk(target):
        filtered = filter(on_filter, filenames)
        result.extend((dirpath, filename) for filename in filtered)
    return result

def assert_utf8(filename:str):
    with open(filename, 'r', encoding='utf-8') as f:
        f.readline()