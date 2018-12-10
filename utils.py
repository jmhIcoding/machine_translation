__author__ = 'jmh081701'
import  json
import  copy
import  numpy as np
import  random

class  DATAPROCESS:
    def __init__(self,source_ling_path,dest_ling_path,source_word_embedings_path,source_vocb_path,dest_word_embeddings_path,dest_vocb_path,seperate_rate=0.05,batch_size=100):
        self.src_data_path =source_ling_path #源语
        self.dst_data_path =dest_ling_path  #目标语对应翻译结果

        self.src_word_embedding_path = source_word_embedings_path #中文预训练的词向量
        self.src_vocb_path  = source_vocb_path   #预训练好的中文词典

        self.dst_word_embedding_path=dest_word_embeddings_path #预训练好的 英文单词词向量
        self.dst_vocb_path = dest_vocb_path    #预训练好的英文词典

        self.seperate_rate =seperate_rate       #测试集 训练集 划分比率
        self.batch_size = batch_size
        self.sentence_length = 100              #截断或填充的句子长度,全部统一

        #data structure to build
        self.src_data_raw=[]    #全部数据集
        self.dst_data_raw =[]
        self.src_train_raw=[]   #训练集
        self.dst_train_raw = []
        self.src_test_raw =[]   #测试集
        self.dst_test_raw =[]

        self.src_word_embeddings=None   #中文词 词向量以及词典
        self.src_id2word=None
        self.src_word2id=None
        self.src_embedding_length =0

        self.dst_word_embeddings=None   #英文 词向量以及词典
        self.dst_id2word=None
        self.dst_word2id=None
        self.dst_embedding_length =0

        self.__load_wordebedding()


        self.__load_data()

        self.last_batch=0
    def __load_wordebedding(self):
        self.src_word_embeddings=np.load(self.src_word_embedding_path)
        self.embedding_length = np.shape(self.src_word_embeddings)[-1]
        with open(self.src_vocb_path,encoding="utf8") as fp:
            self.src_id2word = json.load(fp)
        self.src_word2id={}
        for each in self.src_id2word:
            self.src_word2id.setdefault(self.src_id2word[each],each)

        self.dst_word_embeddings=np.load(self.dst_word_embedding_path)
        self.embedding_length = np.shape(self.dst_word_embeddings)[-1]
        with open(self.dst_vocb_path,encoding="utf8") as fp:
            self.dst_id2word = json.load(fp)
        self.dst_word2id={}
        for each in self.dst_id2word:
            self.dst_word2id.setdefault(self.dst_id2word[each],each)

    def __load_data(self):

        with open(self.src_data_path,encoding='utf8') as fp:
            train_data_rawlines=fp.readlines()
        with open(self.dst_data_path,encoding='utf8') as fp:
            train_label_rawlines=fp.readlines()
        total_lines = len(train_data_rawlines)
        assert len(train_data_rawlines)==len(train_label_rawlines)

        for index in range(total_lines):
            data_line = train_data_rawlines[index].split(" ")[:-1]
            label_line = train_label_rawlines[index].split(" ")[:-1]

            #add and seperate valid ,train set.
            data=[int(self.src_word2id.get(each,0)) for each in data_line]
            label=[int(self.dst_word2id.get(each,0)) for each in label_line]

            self.src_data_raw.append(data)
            self.dst_data_raw.append(label)

            if random.uniform(0,1) <self.seperate_rate:
                self.src_test_raw.append(data)
                self.dst_test_raw.append(label)
            else:
                self.src_train_raw.append(data)
                self.dst_train_raw.append(label)

        self.train_batches= [i for i in range(int(len(self.src_train_raw)/self.batch_size) -1)]
        self.train_batch_index =0

    def pad_sequence(self,sequence,object_length,pad_value=None):
        '''
        :param sequence: 待填充的序列
        :param object_length:  填充的目标长度
        :return:
        '''
        sequence =copy.deepcopy(sequence)
        if pad_value is None:
            sequence = sequence*(1+int((0.5+object_length)/(len(sequence))))
            sequence = sequence[:object_length]
        else:
            sequence = sequence+[pad_value]*(object_length- len(sequence))
        return sequence

    def next_train_batch(self):
        #padding
        output_x=[]
        output_label=[]
        efficient_sequence_length=[]
        index =self.train_batches[self.train_batch_index]
        self.train_batch_index =(self.train_batch_index +1 ) % len(self.train_batches)
        datas = self.src_train_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.dst_train_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length)
            label = self.pad_sequence(labels[index],self.sentence_length)
            output_x.append(data)
            output_label.append(label)
            efficient_sequence_length.append(min(100,len(labels[index])))
        return output_x,output_label,efficient_sequence_length
        #返回的都是下标,注意efficient_sequence_length是有效的长度

    def test_data(self):
        output_x=[]
        output_label=[]
        efficient_sequence_length=[]
        datas = self.src_test_raw[0:]
        labels = self.dst_test_raw[0:]
        for index in range(len(datas)):
            #复制填充
            data= self.pad_sequence(datas[index],self.sentence_length)
            label = self.pad_sequence(labels[index],self.sentence_length)
            output_x.append(data)
            output_label.append(label)
            efficient_sequence_length.append(min(100,len(labels[index])))
        return output_x,output_label,efficient_sequence_length


def evaluate(predict_labels,real_labels,efficient_length):
#输入的单位是batch;
# predict_labels:[batch_size,sequence_length],real_labels:[batch_size,sequence_length]
    sentence_nums =len(predict_labels) #句子的个数
    predict_cnt=0
    predict_right_cnt=0
    real_cnt=0
    for sentence_index in range(sentence_nums):
        try:
            pass
            #predict_set=extract_named_entity(predict_labels[sentence_index],efficient_length[sentence_index])
            #real_set=extract_named_entity(real_labels[sentence_index],efficient_length[sentence_index])
            #right_=predict_set.intersection(real_set)
            #predict_right_cnt+=len(right_)
            #predict_cnt += len(predict_set)
            #real_cnt +=len(real_set)
        except Exception as exp:
            print(predict_labels[sentence_index])
            print(real_labels[sentence_index])
    precision = predict_right_cnt/(predict_cnt+0.000000000001)
    recall = predict_right_cnt/(real_cnt+0.000000000001)
    F1 = 2 * precision*recall/(precision+recall+0.00000000001)
    return {'precision':precision,'recall':recall,'F1':F1}

if __name__ == '__main__':
    dataGen = DATAPROCESS(source_ling_path="data/cn.txt",
                          dest_ling_path="data/en.txt",
                          source_word_embedings_path="data/cn.txt.ebd.npy",
                          source_vocb_path="data/cn.txt.vab",
                          dest_word_embeddings_path="data/en.txt.ebd.npy",
                          dest_vocb_path="data/en.txt.vab",
                          batch_size=3,
                          seperate_rate=0.1
                        )
    src,dst,length=dataGen.next_train_batch()
    print(src)
    print(dst)
    print(length)

