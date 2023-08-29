import tushare as ts
import json
import random
import datetime
import load
from mail import *
# 设置tushare的token，需要先在tushare官网注册并获取
load.initconfig()
ftoken = open("token", "r")
token = ftoken.readline().strip()
ts.set_token(token)
ftoken.close()
flog =  open('log.txt', 'a')

# 初始化tushare的pro接口
pro = ts.pro_api()

# 打开JSON文件并读取内容
with open('config.json', 'r' ,encoding='utf-8') as f:
    json_data = json.load(f)

# 遍历股票代码及其买入点和卖出点
log = ""
# 获取今天的日期
today = datetime.datetime.today()
# 将日期按照指定的格式进行展示
date_str = today.strftime("%Y%m%d")
for stock_code, points in json_data.items():
    print("股票代码：", stock_code)
    name = str(points["name"])
    df = pro.daily(ts_code=stock_code, start_date=date_str, end_date=date_str)
    if(df.shape[0] == 0):
        closeprice = 1000000
        json_data[stock_code]["closeprice"] = closeprice
        continue
    closeprice = float(df.iloc[0][5])
    #closeprice = random.randint(0,100)
    json_data[stock_code]["closeprice"] = closeprice

def sort_function(kv_pair):
    points = kv_pair[1]
    buypoint = float(points["buypoint"])
    sellpoint = float(points["sellpoint"])
    holdstate = int(points["hold_state"])
    closeprice = float(points["closeprice"])
    rate = abs(0.5 - ((closeprice-buypoint)/(sellpoint-buypoint)))
    if holdstate == 1:
        rate += 10000
    return -rate

sorted_data = sorted(json_data.items(), key=sort_function)

print(sorted_data)
for stock_code, points in sorted_data:
    name = str(points["name"])
    jingzichan = float(points["jingzichan"])
    roe_low = float(points["roe_low"])
    roe_high = float(points["roe_high"])
    youzhi_rate = float(points["youzhi_rate"])
    buypoint = float(jingzichan * ((1+(roe_low/100))**5)) - jingzichan*(1-youzhi_rate)
    sellpoint = float(jingzichan * ((1+(roe_high/100))**5)) - jingzichan*(1-youzhi_rate)
    holdstate = str(points["hold_state"])
    closeprice = float(points["closeprice"])
    guxirate = 100*(float(points["guxi"]/float(points["closeprice"])))
    log += name + "\nhold state:" + holdstate + '\n'
    if(closeprice < buypoint):
        log += stock_code + " should be buy\n"
    if(closeprice > sellpoint):
        log += stock_code + " should be sell\n"
    rate = (closeprice-buypoint)/(sellpoint-buypoint)
    log += stock_code + " rate:" + '{:.3f}'.format(rate) + " buypoint:" + '{:.3f}'.format(buypoint) + " sellpoint: " + '{:.3f}'.format(sellpoint) + "\n"
    log += "guxi rate:" + '{:.3f}'.format(guxirate) + '\n'
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
