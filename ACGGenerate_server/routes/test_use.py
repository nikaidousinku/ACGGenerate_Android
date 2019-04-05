import datetime
import time
Character = {'hair color': 'random', 'hair style': 'random', 'eyes color': 'random', 'blush': 'random',
             'smile': 'random', 'open mouth': 'random', 'hat': 'random', 'ribbon': 'random', 'glasses': 'random'}
temp = b'hair color:blue,hair style:random,eyes color:random,blush:random,smile:random,open mouth:random,hat:random,ribbon:random,glasses:random'
temp2 = temp.decode('utf-8')
temp3 = temp2.split(",")
Character['hair color']=temp3[0].split(':')[-1]
Character['hair style']=temp3[1].split(':')[-1]
Character['eyes color']=temp3[2].split(':')[-1]
Character['blush']=temp3[3].split(':')[-1]
Character['smile']=temp3[4].split(':')[-1]
Character['open mouth']=temp3[5].split(':')[-1]
Character['hat']=temp3[6].split(':')[-1]
Character['ribbon']=temp3[7].split(':')[-1]
Character['glasses']=temp3[8].split(':')[-1]
print(Character['hair color'])
now_time=datetime.datetime.now()
now_time=now_time.strftime('%Y-%m-%d %H:%M:%S')
print(now_time)
# print(temp3)
# print(type(temp),type(temp2),type(temp3))
