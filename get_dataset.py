import re
import urllib
import urllib.request
import sys
from tqdm import tqdm
import requests
import os

root = '/work3/s212645/Spectral_Reconstruction/CAVE/data/mat'
if not os.path.exists(root):
    os.makedirs(root)
def download(path, url):
    req = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            f.write(chunk)

#获取页面源码
def getHtml(url):
     page = urllib.request.urlopen(url)   # 打开页面
     html = page.read()       #获取目标页面的源码
     html = html.decode('utf-8')
     return html

# url = 'http://icvl.cs.bgu.ac.il/hyperspectral'
url = 'https://www.cs.columbia.edu/CAVE/databases/multispectral/zip'
html = getHtml(url)

# reg = re.compile(r'https://.*.mat')  
reg = re.compile(r'href=\".*.zip">')  
imglist = re.findall(reg,html)   #解析页面源码获取图片列表

# imglist = imglist[0:len(imglist):2]
x = 0
for imgurl in tqdm(imglist):
    # print(imgurl.split('\"')[1])
    fileurl = '/'+ imgurl.split('\"')[1]
    try:
        path = root + fileurl
        # print(path)
        download_url = url + fileurl
        print(download_url)
        download(path=path, url=download_url)
    except:
        print('Unexpected error:',sys.exc_info())
