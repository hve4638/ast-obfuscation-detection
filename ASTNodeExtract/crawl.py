import csv
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Final

SEARCH_URL:Final = 'https://api.github.com/search/repositories?q={}&page={}'

def search_repositories(query:str, page:int=1)->Dict:
    response = requests.get(SEARCH_URL.format(query, page))
    if response.status_code == 200:
        data = response.json()
        print(data['total_count'])
        for item in data['items']:
            print(item['name'])

if __name__ == '__main__':
    with open('data.csv', 'w') as f:
        items = search_repositories('language:powershell', 1)
        writer = csv.writer(f)