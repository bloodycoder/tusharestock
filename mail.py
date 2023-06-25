import smtplib
import base64
from email.mime.text import MIMEText
from email.header import Header
class PicardSendMail:
    '''
    usage:
    robot = PicardSendMail()
    robot.Login()
    robot.SendAnEmail(email="hello",subject="hello",destination="510297127@qq.com")
    robot.Quit()
    '''
    def __init__(self):
        self.encoding = 'utf-8'
        self.SMTPServer = "smtp.qq.com"
        self.sender = "510297127@qq.com"
        f = open("token", "r")
        f.readline()
        passwd = f.readline().strip()
        print(passwd)
        self.password = passwd
        self.qqMailPort = 587
    def Login(self):
        self.mailserver = smtplib.SMTP(self.SMTPServer,self.qqMailPort)
        self.mailserver.set_debuglevel(0)
        self.mailserver.starttls()
        self.mailserver.login(self.sender,self.password)
    def SendAnEmail(self,email = "hello from python.",subject="picard xie",destination="picardxie@foxmail.com"):
        mail = MIMEText(email.encode(self.encoding),'plain',self.encoding)
        mail['Subject'] = Header(subject,self.encoding)
        mail['From'] = self.sender
        mail['To'] = destination
        self.mailserver.sendmail(self.sender,destination,mail.as_string())
    def Quit(self):
        self.mailserver.quit()