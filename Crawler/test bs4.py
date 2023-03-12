# 文档解析 Beautiful Soup提供一些简单的、python式的函数用来处理导航、搜索、修改分析树等功能。
# Beautiful Soup将复杂HTML文档转换成一个复杂的树形结构,每个节点都是Python对象,所有对象可以归纳为4种
# （1）tag 标签（2）NavigableString （3）BeautifulSoup （4）Comment
import re

from bs4 import BeautifulSoup

'''
#（1）tag 标签及其内容：拿到它找到的第一个内容
file = open('./baidu.html','rb')
html = file.read()

bs = BeautifulSoup(html,'html.parser')
# print(bs.title)              #<title>百度一下，你就知道</title>
# print(type(bs.title))        # <class 'bs4.element.Tag'>
#
#
# #（2）NavigableString 标签里的内容（字符串）
# print(bs.title.string)       #百度一下，你就知道
# print(type(bs.title.string)) #<class 'bs4.element.NavigableString'>
#
# print(bs.a.attrs)   #标签里所有的属性


# #（3）BeautifulSoup 表示整个文档
# print(type(bs)) #<class 'bs4.BeautifulSoup'>
# print(bs.name)
# print(bs.attrs)
# print(bs)


#（4）Comment 注释,一个特殊的NavigableString，输出的内容不包含注释符号
print(bs.a.string)
print(type(bs.a.string)) #<class 'bs4.element.NavigableString'>

'''
'''
file = open('./baidu.html','rb')
html = file.read().decode('utf-8')
bs = BeautifulSoup(html,'html.parser')

#文档的遍历

# print(bs.head.contents)
print(bs.head.contents[1])

'''

# 文档的搜索
'''
file = open('./baidu.html','rb')
html = file.read().decode('utf-8')
bs = BeautifulSoup(html,'html.parser')

#(1)find_all()
# t_list = bs.find_all('a') #查找出html文档中含a的标签
# print(t_list)

#正则表达式搜索：使用search（）方法来匹配内容
# t_list =  bs.find_all(re.compile('a')) #查找出html文档中含a的所有字样
# print(t_list)

#方法：传入一个函数（方法），根据函数的要求来搜索
def name_is_exists(tag):
    return tag.has_attr('name') #搜索含有name标签

t_list = bs.find_all(name_is_exists)
print(t_list)
'''

'''
#(2) kwargs 参数，指定参数进行搜索
# file = open('./baidu.html','rb')
# html = file.read().decode('utf-8')
# bs = BeautifulSoup(html,'html.parser')

# t_list = bs.find_all(id='head') #head头文件下所有的内容，输出为列表
# t_list = bs.find_all(class_=True)
# print(t_list)
'''

'''
# (3) text 参数，文本参数进行搜索
file = open('./baidu.html', 'rb')
html = file.read().decode('utf-8')
bs = BeautifulSoup(html, 'html.parser')

# t_list = bs.find_all(text='hao123')
# t_list = bs.find_all(text=re.compile('\d'))
'''

'''
# (3) limit 参数

file = open('./baidu.html', 'rb')
html = file.read().decode('utf-8')
bs = BeautifulSoup(html, 'html.parser')

t_list = bs.find_all('a', limit=3) #查找a并且限定个数
for i in t_list:
    print(i)
'''


#CSS 选择器
'''
file = open('./baidu.html', 'rb')
html = file.read().decode('utf-8')
bs = BeautifulSoup(html, 'html.parser')

# t_list = bs.select('title') #通过标签来查找
# t_list = bs.select('.mnav') #通过类名来查找
# t_list = bs.select('#u1') #通过id来查找
# t_list = bs.select('a[class="s-top-login-btn c-btn c-btn-primary c-btn-mini lb"]') #通过属性来查找
# t_list = bs.select('body > div') #通过子标签来查找
# t_list = bs.select('.mnav ~ .bri') #通过兄弟标签来查找（和.mnav在同一梯度下的标签.bri）

'''















