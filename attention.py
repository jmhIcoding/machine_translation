__author__ = 'jmh081701'
import  tensorflow as tf
import sys
import os
sys.path.append(os.path.realpath("../"))
from utils import  DATAPROCESS

dataGen = DATAPROCESS(source_ling_path="data/cn.txt",
                          dest_ling_path="data/en.txt",
                          source_word_embedings_path="data/cn.txt.ebd.npy",
                          source_vocb_path="data/cn.txt.vab",
                          dest_word_embeddings_path="data/en.txt.ebd.npy",
                          dest_vocb_path="data/en.txt.vab",
                          batch_size=5,
                          seperate_rate=0.1
                        )
dataGen.src_word_embeddings
class Model:
    def __init__(self):
        pass
class AttentionModel:
    def __init__(self,src_word_embeddings,dst_word_embeddings,word_embedding_len,lstm_hidden_nums):
        self.src_word_embeddings=src_word_embeddings
        self.dst_word_embeddings=dst_word_embeddings

        self.word_embedding_len=word_embedding_len

        self.lstm_hidden_nums = lstm_hidden_nums
        self.set_placeholder()
        self.set_word_embedding_matrix()
        self.set_score_matrix()


    def set_placeholder(self):
        self.src_sentences=tf.placeholder(dtype=tf.int32,shape=[None,None],name="input-src-sentences")#中文句子,形状应该是batch_size,句子最大填充长度
        self.dst_sentences=tf.placeholder(dtype=tf.int32,shape=[None,None],name="input-dst-sentences")#英文句子,翻译的标准答案,batch_size,句子的最大填充长度
        self.src_real_sentence_lengths=tf.placeholder(dtype=tf.int32,shape=[None],name='src-real-sequences-len')#中文句子的实际有效长度(去除填充的部分)
        self.dst_real_sentence_lengths=tf.placeholder(dtype=tf.int32,shape=[None],name='dst-real-sequences-len')

    def set_word_embedding_matrix(self):
        self.src_embedding_matrix=tf.Variable(initial_value=self.src_word_embeddings,trainable=True,name='src_embedding')
        self.dst_embedding_matrix=tf.Variable(initial_value=self.dst_word_embeddings,trainable=True,name='dst-embedding')

    def set_score_matrix(self):
        self.Ua=tf.get_variable(name='score-Ua',shape=[self.word_embedding_len,self.word_embedding_len],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.Wa=tf.get_variable(name='score-Wa',shape=[self.word_embedding_len,self.word_embedding_len],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.Va=tf.get_variable(name='score-Va',shape=[self.word_embedding_len,1],initializer=tf.contrib.layers.xavier_initializer())
    def build_graph(self):
        #encoder部分
        src_data=tf.nn.embedding_lookup(self.src_embedding_matrix,ids=self.src_sentences)
        self.fw_lstm_encoder=tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums,activation=tf.nn.relu)
        self.bw_lstm_encoder=tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums,activation=tf.nn.relu)

        output,_=tf.nn.bidirectional_dynamic_rnn(self.fw_lstm_encoder,self.bw_lstm_encoder,inputs=src_data,sequence_length=self.src_real_sentence_lengths)

        fw_output=output[0]#batch_size,max_sequence,hidden_nums
        bw_output=output[1]#batch_size,max_sequence,hidden_nums

        contact=tf.concat([fw_output,bw_output],-1) #batch_size,max_sequence,2*hidden_nums
        shape_before_ = tf.shape(contact)

        contact=tf.reshape(contact,shape=[-1,2*self.lstm_hidden_nums])
        Wc_encoder=tf.get_variable(name='Wc-encoder',shape=[2*self.lstm_hidden_nums,self.lstm_hidden_nums],initializer=tf.contrib.layers.xavier_initializer)
        contact=tf.tanh(tf.matmul(contact,Wc_encoder))
        hiddens=tf.reshape(contact,shape=[shape_before_[0],shape_before_[1],self.lstm_hidden_nums],name='lstm-hidden-layers')


        #Decoder






