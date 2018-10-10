import requests
import re
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import tarfile

def download(dir):
    '''
    Downloads data, and returns the downloaded filename
    '''
    url = 'https://www.cs.toronto.edu/~kriz/cifar.html'
    response = requests.get(url, stream=True)

    soup = BeautifulSoup(response.content,'html5lib')
    links = soup.findAll('a', attrs={'href': re.compile("cifar-10")})

    filename = str(links[0]).split('"')[1]
    url = url.split('cifar')[0]+filename

    with open(os.path.join(dir,filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    return os.path.join(dir,filename)

def untar(filename):
    '''
    Unpacks given filename
    '''
    tar = tarfile.open(filename)
    tar.extractall(path=dir)
    tar.close()

dir = 'datasets'

if not os.path.exists(dir):
    os.mkdir(dir)

filename = download(dir)
untar(filename)
