import hashlib
import random
import time
import queue
import requests
import sys, os
import json
from threading import Thread

URL_VT_SCAN = 'https://www.virustotal.com/vtapi/v2/file/scan'
URL_VT_REPORT = 'https://www.virustotal.com/vtapi/v2/file/report'
def upload_vt(filename, api_key):
    scan = __request_scan(filename, api_key)
    if scan is None:
        return None
    
    resource = scan['resource']
    time.sleep(30)
    retires = 5

    while True:
        report = __request_report(resource, api_key)
        if report['response_code'] == 1:
            break
        else:
            time.sleep(60)
            retires -= 1
            if retires < 0:
                return None
    
    return report

def __request_scan(filename:str, api_key:str, retries:int=5):
    if retries < 0:
        return None
    
    with open(filename, 'rb') as file_obj:
        files = {'file': (filename, file_obj) }

        params = {'apikey': api_key }
        res = requests.post(URL_VT_SCAN, files=files, params=params)
        match res.status_code:
            case 200:
                return res.json()
            case 204:
                sys.stderr.write('API rate limit exceeded. Wait 1 minute.\n')
                time.sleep(60)
                return __request_scan(filename, api_key, retries-1)
            case _:
                print('Error:', res.status_code)
                print(res)
                raise 'STOP'

def __request_report(resource:str, api_key:str, retries:int=5):
    if retries < 0:
        return None
    
    params = {'apikey': api_key, 'resource': resource }
    res = requests.get(URL_VT_REPORT, params=params )
    match res.status_code:
        case 200:
            return res.json()
        case 204:
            sys.stderr.write('API rate limit exceeded. Wait 1 minute.\n')
            time.sleep(60)
            return __request_report(resource, api_key, retries-1)
        case _:
            print('Error:', res.status_code)
            print(res)
            raise 'STOP'

