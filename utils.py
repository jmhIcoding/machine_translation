__author__ = 'jmh081701'
import  json
import  copy
import  numpy as np
import  random

from src.df.src.utility import LoadDataWalkieTalkieCW
from src.df.src.utility import Cluster_consecutive_bursts

class  DATAPROCESS:
    def __init__(self,
                 seperate_rate=0.05,
                 batch_size=100,
                 observe_length=10,
                 vocb_size=9520,
                 is_test=False
    ):

        self.obv_length = observe_length
        self.seperate_rate =seperate_rate       #验证集占比
        self.batch_size = batch_size
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

        self.src_cdf=[]                         #源语 句子长度的累计分布 src_cdf[t]=0.6,表示 P({x<=t}) = 60%
        self.dst_cdf=[]                         #目标语 句子长度的累计分布 dst_cdf[t]=0.6,表示 P({x<=t}) = 60%

        self.src_sentence_length = 40           #encode 输入序列的长度
        self.dst_sentence_length = 40           #decode 输出结果序列的长度

        self.last_batch=0
        self.epoch =0
        self.vocb_size = vocb_size              #每个burst_size 属于有正也有负

        self.start_size_id   = vocb_size -1     #起始标识
        self.end_size_id     = 0                #结束标注
        self.end_token_id    = self.size_to_tokenid(self.end_size_id)
        self.start_token_id = self.size_to_tokenid(self.start_size_id)

        self.__load_wordebedding()
        self.__load_data(is_test=is_test)

    def size_to_tokenid(self,burst_size):
        #把负方向的busrt size 映射到4758维以上
        if burst_size < 0:
            return self.vocb_size//2 - burst_size
        else:
            return burst_size
    def tokenid_to_size(self,tokenid):
        if tokenid >self.vocb_size//2:
            return  self.vocb_size//2 - tokenid
        else:
            return tokenid
    def __load_wordebedding(self):
        print('There is no need to load pretrained  embedding wordvector')
        pass
    def cdf(self,length_list,percentile):
        return  np.percentile(length_list,percentile)
    def _real_length(self,x):
        ##x.shape:(1,2300,1)
        x   = np.array(x)
        l= 0
        r = x.shape[0]
        while l < r:
            mid = (l+r)//2
            if abs(x[mid])> 0:
                l =mid+1
            else:
                r = mid
        return l
    def __load_data(self,is_test = False):

        X_train, y_train, X_valid, y_valid, X_test, y_test=LoadDataWalkieTalkieCW(is_cluster=False)
        if not is_test:
            X=np.concatenate([X_train,X_valid])
        else:
            X=X_test

        raw_text = Cluster_consecutive_bursts(X,normalized=False,padding=True,return_array=False)
        #每一行就是一个句子,而且是填充好的.

        train_data_rawlines = copy.deepcopy(raw_text)
        train_label_rawlines = copy.deepcopy(raw_text)
        del raw_text

        total_lines = len(train_data_rawlines)
        assert len(train_data_rawlines)==len(train_label_rawlines)
        src_len=[]
        dst_len=[]
        for index in range(total_lines):
            #add and seperate valid ,train set.
            data= [self.start_size_id] + train_label_rawlines[index]
            data =data[:self.src_sentence_length]           #把起始的start_burst_size_id加上,然后截断为固定的长度

            if self._real_length(data) <= self.obv_length:
                #过滤掉小于观察长度的数据
                continue
            data=[self.size_to_tokenid(x) for x  in data]   #把burst_size_id转换为token的表示方法,
            #print('full data:',data)
            label  = copy.deepcopy(data[self.obv_length:])                     #这是输入到decoder的
            #print('label:',label)
            src_len.append(self._real_length(data))
            dst_len.append(self._real_length(label))
            #
            data=data[:self.obv_length] #+[self.end_token_id]*(self.src_sentence_length-self.obv_length)      #截断!只保留[0,obv_length),obv_length之后的内容全部为0，这是输入为encoder的
            #print('obv:',data)
            self.src_data_raw.append(data)
            self.dst_data_raw.append(label)

            if random.uniform(0,1) <self.seperate_rate:
                self.src_test_raw.append(data)
                self.dst_test_raw.append(label)
            else:
                self.src_train_raw.append(data)
                self.dst_train_raw.append(label)

        self.src_len_std=np.std(src_len)
        self.src_len_mean=np.mean(src_len)
        self.src_len_max=np.max(src_len)
        self.src_len_min=np.min(src_len)
        self.src_cdf = self.cdf(src_len,[i for i in range(1,100,1)])

        self.dst_len_std=np.std(dst_len)
        self.dst_len_mean=np.mean(dst_len)
        self.dst_len_max = np.max(dst_len)
        self.dst_len_min=np.min(dst_len)
        self.dst_cdf = self.cdf(dst_len,[i for i in range(1,100,1)])

        self.train_batches= [i for i in range(int(len(self.src_train_raw)/self.batch_size) -1)]
        self.train_batch_index = 0

        self.test_batches= [i for i in range(int(len(self.src_test_raw)/self.batch_size) -1)]
        self.test_batch_index = 0

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
        return sequence[:object_length]

    def next_train_batch(self):
        #padding
        output_x=[]
        output_label=[]
        src_sequence_length=[]
        dst_sequence_length=[]
        index =self.train_batches[self.train_batch_index]
        self.train_batch_index =(self.train_batch_index +1 ) % len(self.train_batches)
        if self.train_batch_index is 0:
            self.epoch +=1
        datas = self.src_train_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.dst_train_raw[index*self.batch_size:(index+1)*self.batch_size]

        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.src_sentence_length,pad_value=0)    #源语
            #label = labels[index]
            label = self.pad_sequence(labels[index],self.dst_sentence_length,pad_value=0) #目标语
            label[-1] = 0
            output_x.append(data)
            output_label.append(label)

            src_sequence_length.append(self._real_length(datas[index]))         #真实长度
            dst_sequence_length.append(len(labels[index]))    #真实长度，注意要把end-token-加入到里面去
        return output_x,output_label,src_sequence_length,dst_sequence_length
        #返回的都是下标,注意src(dst)_sequence_length是有效的长度
    def next_test_batch(self):
        output_x=[]
        output_label=[]
        src_sequence_length=[]
        dst_sequence_length=[]
        index =self.test_batches[self.test_batch_index]
        self.test_batch_index =(self.test_batch_index +1 ) % len(self.test_batches)
        datas = self.src_test_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.dst_test_raw[index*self.batch_size:(index+1)*self.batch_size]

        for index in range(len(datas)):
            #复制填充
            data= self.pad_sequence(datas[index],self.src_sentence_length,pad_value=0)
            label = self.pad_sequence(labels[index],self.dst_sentence_length,pad_value=0)
            label[-1] =0

            output_x.append(data)
            output_label.append(label)
            src_sequence_length.append(self._real_length(datas[index]))
            dst_sequence_length.append(len(self._real_length(labels[index])))
        return output_x,output_label,src_sequence_length,dst_sequence_length
    def test_data(self):
        output_x=[]
        output_label=[]
        src_sequence_length=[]
        dst_sequence_length=[]
        datas = self.src_test_raw[0:]
        labels = self.dst_test_raw[0:]
        for index in range(len(datas)):
            #复制填充
            data= self.pad_sequence(datas[index],self.src_sentence_length,pad_value=0)
            label = self.pad_sequence(labels[index],self.dst_sentence_length,pad_value=0)
            label[-1] =0
            output_x.append(data)
            output_label.append(label)
            src_sequence_length.append(self._real_length(datas[index]))
            dst_sequence_length.append(self._real_length(labels[index]))
        start=0
        end=len(datas)
        while len(output_x)< self.batch_size:
            #不满一个batch就填充
            output_x.append(output_x[start])
            output_label.append(output_label[start])
            src_sequence_length.append(src_sequence_length[start])
            dst_sequence_length.append(dst_sequence_length[start])
            start=(start+1) % end
        return output_x,output_label,src_sequence_length,dst_sequence_length

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
    batch_size = 5
    dataGen = DATAPROCESS(
                          batch_size=batch_size,
                          seperate_rate=0.1
                        )
    print("-"*10+"src corpus"+'-'*20)
    print({'std':dataGen.src_len_std,'mean':dataGen.src_len_mean,'max':dataGen.src_len_max,'min':dataGen.src_len_min})

    print('-'*10+"dst corpus"+'-'*20)
    print({'std':dataGen.dst_len_std,'mean':dataGen.dst_len_mean,'max':dataGen.dst_len_max,'min':dataGen.dst_len_min})
    print("#"*30)
    print("source corpus percentile")
    src_cdf=dataGen.src_cdf
    for i in range(len(src_cdf)):
        print(i,src_cdf[i])

    print("#"*30)
    print("destination corpus percentile ")
    dst_cdf=dataGen.dst_cdf
    for i in range(len(dst_cdf)):
        print(i,dst_cdf[i])
    '''
            ----------src corpus--------------------
            {'std': 0.0, 'min': 50, 'mean': 50.0, 'max': 50}
            ----------dst corpus--------------------
            {'std': 272.76550908345234, 'min': 51, 'mean': 346.51727982162765, 'max': 2231}
    '''
    output_x,output_label,src_sequence_length,dst_sequence_length = dataGen.next_train_batch()
    print(src_sequence_length)
    print(dst_sequence_length)
    for i in range(batch_size):
        for j in range(len(output_x[i])):
            print("{2},real:{0},label:{1}".format(output_x[i][j],output_label[i][j],j))
        print("########################################")

