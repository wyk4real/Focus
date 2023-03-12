import urllib.request

# 获取一个get请求
# response = urllib.request.urlopen('https://www.baidu.com')
# print(response.read().decode('utf-8')) #对获取到的网站进行utf-8解码

# 获取一个post请求
# import urllib.parse
# date = bytes(urllib.parse.urlencode({'hallo':'world'}),encoding='utf-8')
# response = urllib.request.urlopen('https://httpbin.org/post',date)
# print(response.read().decode('utf-8'))

#超时处理：如果超过给定的时间结束此网站的爬虫(timeout=...) 使用try
# try:
#     response = urllib.request.urlopen('https://httpbin.org/get',timeout=0.01)
#     print(response.read().decode('utf-8'))
# except urllib.error.URLError as e:
#     print('time out!')


# response = urllib.request.urlopen('https://baidu.com')
# print(response.status)
# print(response.getheaders())

'''
url = 'https://httpbin.org/post'
headers = {
'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
}
data = bytes(urllib.parse.urlencode({'name':'eric'}),encoding = 'utf-8')
req = urllib.request.Request(url=url,data=data,headers=headers,method='POST')
response = urllib.request.urlopen(req)
print(response.read().decode('utf-8'))
'''




url = 'https://douban.com'
headers = {
'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
}

req = urllib.request.Request(url=url,headers=headers)
response = urllib.request.urlopen(req)
print(response.read().decode('utf-8'))




