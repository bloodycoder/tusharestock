import tushare as ts
import json
import datetime
from mail import *
# 设置tushare的token，需要先在tushare官网注册并获取
ftoken = open("token", "r")
token = ftoken.readline().strip()
ts.set_token(token)
ftoken.close()
flog =  open('log.txt', 'a')

# 初始化tushare的pro接口
pro = ts.pro_api()

# 打开JSON文件并读取内容
with open('config.json', 'r') as f:
    json_data = json.load(f)

# 遍历股票代码及其买入点和卖出点
log = ""
# 获取今天的日期
today = datetime.datetime.today()
# 将日期按照指定的格式进行展示
date_str = today.strftime("%Y%m%d")
for stock_code, points in json_data.items():
    print("股票代码：", stock_code)
    buypoint = float(points["buypoint"])
    sellpoint = float(points["sellpoint"])
    holdstate = str(points["hold_state"])
    name = str(points["name"])
    df = pro.daily(ts_code=stock_code, start_date=date_str, end_date=date_str)
    print(df)
    closeprice = float(df.iloc[0][5])
    print(closeprice)
    log += name + "\nhold state:" + holdstate + '\n'
    if(closeprice < buypoint):
        log += stock_code + " should be buy\n"
    if(closeprice > sellpoint):
        log += stock_code + " should be sell\n"
    rate = (closeprice-buypoint)/(sellpoint-buypoint)
    log += stock_code + " rate:" + '{:.3f}'.format(rate) + " buypoint:" + str(buypoint) + " sellpoint: " + str(sellpoint) + "\n"
    log += "-------------------------\n"
flog.write(log)
robot = PicardSendMail()
robot.Login()
title = "today stock info:" + date_str
robot.SendAnEmail(email=log,subject=title,destination="510297127@qq.com")
robot.SendAnEmail(email=log,subject=title,destination="2409029740@qq.com")
robot.Quit()
# 调用get_realtime_quotes()函数获取实时行情数据
flog.close()
