#正则表达式：字符串模式（判断字符串是否符合一定的标准）

import re

'''
#创建模式对象
pat = re.compile('AA') #此处的AA是正则表达式，用来去验证其他的字符串
m = pat.search('ABCAA') #用search字符串被校验的内容
print(m) #<re.Match object; span=(3, 5), match='AA'> 判断是否匹配 输出匹配位置（输出符合的第一个位置）

#没有模式对象
n = re.search('asd', 'Aasd') #第一个参数为规则，第二个参数是校验对象
print(n)
'''

'''
m = re.findall('a', 'ASDaDFGAa')
n = re.findall('[A-Z]', 'ASDaDFGAa')
x = re.findall('[A-Z]+', 'ASDaDFGAa')
print(m)  #['a', 'a']
print(n)  #['A', 'S', 'D', 'D', 'F', 'G', 'A']
print(x)  #['ASD', 'DFGA']
'''

'''
#建议在正则表达式中，被比较的字符串前面加r，不要担心转义字符的问题
m = re.sub('a', 'A', r'abcdcasd') #找到a用A替换,
print(m)  #AbcdcAsd
'''













