#-*- coding:utf-8 -*-
#前一步生成的问答文件路径
train_encode_file = 'train.enc'
train_decode_file = 'train.dec'
test_encode_file = 'test.enc'
test_decode_file = 'test.dec'
print("开始创建词汇表....")
#特殊标记，用来填充标记对话
PAD = '_PAD_'
GO = '_GO'
EOS = '_EOS_'  #对话结束
UNK = '_UNK_'  #标记未出现在词汇表中的字符
START_VOVABULARY = [PAD,GO,EOS,UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

#参看tensorflow.models.rnn.translate.data_utils
vocabulary_size = 5000
out_word_list = [",", "?", ".", "。", "!", "...", "？", "，","0","——"]
#生成词汇表文件
def gen_vocabulary_file(input_file,output_file):
    vocabulary = {}
    with open(input_file) as f:
       # counter = 0
        for line in f:
           # counter +=1
            tokens=[word for word in line.strip()]
            for word in tokens:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
        vocabulary_list = START_VOVABULARY + sorted(vocabulary,key=vocabulary.get,reverse=True)
         #取前5000个汉字
    if len(vocabulary_list) > 5000:
        vocabulary_list = vocabulary_list[:5000]
        print(input_file + " 词汇表大小：",len(vocabulary_list))
    with open(output_file,'w') as ff:
        for word in vocabulary_list:
            #print(word)
            ff.write((word+'\n'))

gen_vocabulary_file(train_encode_file,"train_encode_vocabulary")
gen_vocabulary_file(train_decode_file,"train_decode_vocabulary")
