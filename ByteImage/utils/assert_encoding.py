
def assert_utf8(filename:str):
    with open(filename, 'r', encoding='utf-8') as f:
        f.readline()

def assert_utf16le(filename:str):
    with open(filename, 'r', encoding='utf-16le') as f:
        f.readline()

def assert_utf16be(filename:str):
    with open(filename, 'r', encoding='utf-16be') as f:
        f.readline()

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
    