
def assert_utf8(filename:str):
    with open(filename, 'r', encoding='utf-8') as f:
        f.read()

def assert_utf16le(filename:str):
    with open(filename, 'r', encoding='utf-16le') as f:
        f.read()

def assert_utf16be(filename:str):
    with open(filename, 'r', encoding='utf-16be') as f:
        f.read()
    
def assert_cp949(filename:str):
    with open(filename, 'r', encoding='cp949') as f:
        f.read()

def is_utf8(filename:str)->bool:
    try:
        assert_utf8(filename)
        return True
    except Exception as e:
        return False

def is_utf16le(filename:str)->bool:
    try:
        assert_utf16le(filename)
        return True
    except Exception as e:
        return False
    
def is_utf16be(filename:str)->bool:
    try:
        assert_utf16be(filename)
        return True
    except Exception as e:
        return False

def is_cp949(filename:str)->bool:
    try:
        assert_cp949(filename)
        return True
    except Exception as e:
        return False

def try_read(filename:str, encoding:str)->str:
    try:
        with open(filename, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        return None

def assert_utf16le(filename:str):
    with open(filename, 'r', encoding='utf-16le') as f:
        f.read()

def assert_utf16be(filename:str):
    with open(filename, 'r', encoding='utf-16be') as f:
        f.read()
    
def assert_cp949(filename:str):
    with open(filename, 'r', encoding='cp949') as f:
        f.read()