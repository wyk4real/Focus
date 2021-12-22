import re
import urllib.error
import urllib.request
from bs4 import BeautifulSoup
import xlwt


def main():
    baseurl = 'https://movie.douban.com/top250?start='
    # 1. 爬取网页
    datalist = getData(baseurl)
    savepath = '豆瓣电影Top250.xls' #保存在当前路径下
    # 3.保存数据
    saveData(datalist, savepath)


findLink = re.compile(r'<a href="(.*?)">') #创建正则表达式对象（超链接的形式），表示规则（字符串模式）
findImgSrc = re.compile(r'<img.*src="(.*?)"', re.S) #re.S 让换行符包含在字符中
findTitle = re.compile(r'<span class="title">(.*)</span>')
findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
findJudge = re.compile(r'<span>(\d*)人评价</span>')
findInq = re.compile(r'<span class="inq">(.*)</span>')
findBd = re.compile(r'<p class="">(.*?)</p>', re.S)


def getData(baseurl):
    datalist = []
    for i in range(0, 10):
        url = baseurl + str(i*25)
        html = askURL(url)  #获取页面，调用下面askURL函数

        # 2.逐一解析数据
        soup = BeautifulSoup(html, 'html.parser')
        for item in soup.find_all('div', class_="item"): #查找符合要求的字符串，形成列表
            date = [] #保存一步电影的所有信息
            item = str(item)

            #获取影片的详情超链接
            link = re.findall(findLink, item)[0]  #re库用来通过正则表达式查找指定的字符串
            date.append(link)

            img = re.findall(findImgSrc, item)[0]
            date.append(img)

            titles = re.findall(findTitle, item)
            if len(titles) == 2:
                date.append(titles[0])
                date.append(titles[1].replace('/', '')) #删除外文名前面的/
            else:
                date.append(titles[0])
                date.append(' ') #如果没有外文名，则输入为空

            rating = re.findall(findRating, item)[0]
            date.append(rating)

            judge = re.findall(findJudge, item)[0]
            date.append(judge)

            inq = re.findall(findInq, item)
            if len(inq) != 0:
                date.append(inq[0].replace('。', ''))
            else:
                date.append(' ')

            bd = re.findall(findBd, item)[0]
            bd = re.sub('<br(\s+)?/>(\s+)?', " ", bd)  # 去掉<br/>,用空格替换
            bd = re.sub('/', ' ', bd)
            date.append(bd.strip())  # 去掉前后的空格位置

            datalist.append(date)

    return datalist


def askURL(url): #得到指定一个URL的网页内容
    head = {      #模拟浏览器头部信息，向豆瓣服务器发送消息，伪装
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
    } #用户代理，表示告诉豆瓣服务器，我们是什么类型的机器，浏览器（本质上是告诉浏览器，我们可以接受什么水平的文件）
    request = urllib.request.Request(url,headers=head)
    html = ''
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode('utf-8')
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, 'code'):
            print(e.code)
        if hasattr(e, 'reason'):
            print(e.reason)

    return html


def saveData(datelist, savepath):
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('豆瓣电影Top250', cell_overwrite_ok=True)
    col = ('电影详情链接', '图片链接', '影片中文名', '影片外文名', '评分', '评价数', '概况', '相关信息')
    for i in range(8):
        sheet.write(0, i, col[i])
    for i in range(250):
        print('第%d条：' % (i+1))
        date = datelist[i]
        for j in range(7):
            sheet.write(i+1, j, date[j])

    book.save(savepath)


if __name__ == "__main__":
    main()








