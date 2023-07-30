import urllib.request
# 指定URL
url = 'https://raw.githubusercontent.com/bloodycoder/tusharestock/main/config.json'
# 打开URL并读取内容
# 写入本地文件
def initconfig():
    response = urllib.request.urlopen(url)
    content = response.read()
    with open('config.json', 'wb') as f:
        f.write(content)