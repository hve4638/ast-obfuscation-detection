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
    resource = scan['resource']
    time.sleep(5)

    while True:
        report = __request_report(resource, api_key)
        print(report)
        if report['response_code'] == 1:
            break
        else:
            time.sleep(15)
    
    result = report.get('scans')
    return report

def __request_scan(filename:str, api_key:str):
    with open(filename, 'rb') as file_obj:
        files = {'file': (filename, file_obj) }

        params = {'apikey': api_key }
        res = requests.post(URL_VT_SCAN, files=files, params=params)
        match res.status_code:
            case 200:
                return res.json()
            case 204:
                sys.stderr.write('API rate limit exceeded. Wait 15 seconds.\n')
                time.sleep(15)
                return __request_scan(filename, api_key)
            case _:
                print('Error:', res.status_code)
                print(res)
                raise 'STOP'
def __request_report(resource:str, api_key:str):
    params = {'apikey': api_key, 'resource': resource }
    res = requests.get(URL_VT_REPORT, params=params )
    match res.status_code:
        case 200:
            return res.json()
        case 204:
            sys.stderr.write('API rate limit exceeded. Wait 15 seconds.\n')
            time.sleep(15)
            return __request_report(resource, api_key)
        case _:
            print('Error:', res.status_code)
            print(res)
            raise 'STOP'


if __name__ == '__main__':
    api_key = os.getenv('VT_API_KEY')
    report = upload_vt('./HEMacro.dll', api_key)
    with open('result.json', 'w') as f:
        json.dump(report, f, indent=4)
    
