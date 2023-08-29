def cal(jinzichan,rate, roe):
    #5 years 
    for i in range(5,11):
        gujia = jinzichan *(roe**i) - jinzichan*(1-rate)
        print("year ",i,":",gujia)
    #10 years
