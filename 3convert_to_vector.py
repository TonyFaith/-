#-*- coding:utf-8 -*-
#前一步生成的问答文件路径
train_encode_file = 'train.enc'
train_decode_file = 'train.dec'
test_encode_file = 'test.enc'
test_decode_file = 'test.dec'
#前一步生成的词汇表路径
train_encode_vocabulary_file = 'train_encode_vocabulary'
train_decode_vocabulary_file = 'train_decode_vocabulary'
UNK_ID = 3
print("对话转向量...")
#把对话转为向量形式
def convert_to_vector(input_file,vocabulary_file,output_file):
    tmp_vocab = []
    with open(vocabulary_file,'r') as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x,y) for (y,x) in enumerate(tmp_vocab)])
        #{’硕‘：3142,’v‘：577,’I’：4789.。。。}
    output_file = open(output_file,'w')
    with open(input_file,'r') as f:
        for line in f:
            line_vec = []
            for words in line.strip():
                line_vec.append(vocab.get(words,UNK_ID)) #dict.get(key,default=None key--字典中要查找的键，default--如果指定键的值不存在时，返回该默认值值。
            output_file.write(" ".join([str(num) for num in line_vec]) + "\n")
    output_file.close()

#注意：使用训练集的问答词汇表来生成训练集和测试集的向量表
convert_to_vector(train_encode_file,train_encode_vocabulary_file,'train_encode.vec')
convert_to_vector(train_decode_file,train_decode_vocabulary_file,"train_decode.vec")
convert_to_vector(test_encode_file,train_encode_vocabulary_file,"test_encode.vec")
convert_to_vector(test_decode_file,train_decode_vocabulary_file,"test_decode.vec")

#生成的train_encode.vec 和 train_decode.vec 用于训练，对应的词汇表是train_encode_vocabulary和train_decode_vocabulary
