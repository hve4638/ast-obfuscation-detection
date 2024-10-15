import os
import json
from .assert_encoding import is_utf8, is_utf16le, is_utf16be, try_read

def read_bytes_as_utf8(filename:str) -> str:
    data = try_read(filename, 'utf-8')\
            or try_read(filename, 'utf-16le')\
            or try_read(filename, 'utf-16be')\
            or try_read(filename, 'cp949')
    if data is None:
        print(f"Failed to read : {filename}")
        raise 'Failed to read'
    return data.encode('utf-8')

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


def read_as_utf8(filename:str)->str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()