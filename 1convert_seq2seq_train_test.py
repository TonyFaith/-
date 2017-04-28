#encoding:utf-8
import re
import os
import random
import string

conv_path = 'dgk_shooter_min.conv'

if not os.path.exists(conv_path):
    print('数据集不存在')
    exit()

#数据格式
'''
E
M 畹/华/吾/侄/
M 你/接/到/这/封/信/的/时/候/
M 不/知/道/大/伯/还/在/不/在/人/世/了/
E
M 咱/们/梅/家/从/你/爷/爷/起/
M 就/一/直/小/心/翼/翼/地/唱/戏/
M 侍/奉/宫/廷/侍/奉/百/姓/
M 从/来/不/曾/遭/此/大/祸/
M 太/后/的/万/寿/节/谁/敢/不/穿/红/
M 就/你/胆/儿/大/
M 唉/这/我/舅/母/出/殡/
M 我/不/敢/穿/红/啊/
M 唉/呦/唉/呦/爷/
M 您/打/得/好/我/该/打/
M 就/因/为/没/穿/红/让/人/赏/咱/一/纸/枷/锁/
M 爷/您/别/给/我/戴/这/纸/枷/锁/呀/
E
M 您/多/打/我/几/下/不/就/得/了/吗/
M 走/
M 这/是/哪/一/出/啊/…/ / /这/是/
M 撕/破/一/点/就/弄/死/你/
M 唉/

'''

convs = []
with open(conv_path) as f:
    one_dialoge = []
    for line in f :
        line = line.strip('\n').replace('/','')
        if line == '':
            continue
        if line[0] == 'E':
            if one_dialoge:
                convs.append(one_dialoge)
                one_dialoge = []
        elif line[0] == 'M':
            line = line.split(' ')[1]
            for e in line:
                if re.match('[0-9A-Za-z]',e):
                    line = line.replace(e,"")
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+","",line)  #去除中文标点符号
            one_dialoge.append(line)

#print(convs[:3])

#v把对话分成问和答

ask = []
response = []
for one_dialoge in convs:
    if len(one_dialoge) == 1:
        continue
    if len(one_dialoge) % 2 != 0: #奇数对话数，转为i偶数对话数
        one_dialoge = one_dialoge[:-1]
    for i in range(len(one_dialoge)):
        if i%2 == 0:
            ask.append(one_dialoge[i])
        else:
            response.append(one_dialoge[i])

print(len(ask),len(response))
print(ask[:3])
print (response[:3])

def convert_seq2seq_files(questions,answers,TESTSET_SIZE=8000):
    #创建文件
    train_enc = open('train.enc','w') #问
    train_dec = open('train.dec','w') #答
    test_enc = open('test.enc','w') #问
    test_dec = open('test.dec','w') #答

    #选择8000数据作为测试数据
    test_index = random.sample([i for i in range(len(questions))],TESTSET_SIZE)
    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+'\n')
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+'\n')
        if i % 1000 == 0:
            print (len(questions),'处理进度：',i)
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()

convert_seq2seq_files(ask,response)

