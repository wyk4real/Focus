import xlwt

'''
workbook = xlwt.Workbook(encoding='utf-8') #创建workbook
worksheet = workbook.add_sheet('sheet1') #创建工作表
worksheet.write(0, 0, 'hallo') #写入参数0行0列 写入hallo
worksheet.save('student.xls') #保存数据表
'''


#9*9乘法表写入
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet1')
for i in range(10):
    for j in range(i+1):
        worksheet.write(i, j, '%d * %d = %d' % (i+1, j+1, (i+1)*(j+1)))

workbook.save('mal.xls')











