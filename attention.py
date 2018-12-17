__author__ = 'jmh081701'
import  tensorflow as tf
import sys
import os
import tensorflow.contrib.crf
import copy
sys.path.append(os.path.realpath("../"))
from utils import  DATAPROCESS
from tensorflow.contrib.seq2seq import *

dataGen = DATAPROCESS(source_ling_path="data/cn.txt",
                          dest_ling_path="data/en.txt",
                          source_word_embedings_path="data/cn.txt.ebd.npy",
                          source_vocb_path="data/cn.txt.vab",
                          dest_word_embeddings_path="data/en.txt.ebd.npy",
                          dest_vocb_path="data/en.txt.vab",
                          batch_size=5,
                          seperate_rate=0.1
                      )

class Model:
    def __init__(self):
        pass

class AttentionModel:
    TRAIN=1
    PREDICT=0
    def __init__(self,learning_rate,src_word_embeddings,dst_word_embeddings,word_embedding_len,lstm_hidden_nums,max_sequence,dst_vocb_size,train_flag,batch_size):
        self.src_word_embeddings=src_word_embeddings
        self.dst_word_embeddings=dst_word_embeddings

        self.word_embedding_len=word_embedding_len
        self.sequence_max_length=max_sequence
        self.lstm_hidden_nums = lstm_hidden_nums

        self.set_placeholder()
        self.set_word_embedding_matrix()
        self.set_score_matrix()
        self.dst_vocb_size=dst_vocb_size
        self._dst_real_sentence_lengths_=None
        self._src_real_sentence_lengths_=None
        self.train_flag = train_flag
        self.batch_size = batch_size
        self.lr = learning_rate
    def set_placeholder(self):
        self.src_sentences=tf.placeholder(dtype=tf.int32,shape=[None,None],name="input-src-sentences")#中文句子,形状应该是batch_size,句子最大填充长度
        self.dst_sentences=tf.placeholder(dtype=tf.int32,shape=[None,None],name="input-dst-sentences")#英文句子,翻译的标准答案,batch_size,句子的最大填充长度
        self.src_real_sentence_lengths=tf.placeholder(dtype=tf.int32,shape=[None],name='src-real-sequences-len')#中文句子的实际有效长度(去除填充的部分)
        self.dst_real_sentence_lengths=tf.placeholder(dtype=tf.int32,shape=[None],name='dst-real-sequences-len')

        self.start_tokens =tf.placeholder(dtype=tf.int32,shape=[None,None],name='start-tokens-predict') #每个batch内的每个句子都要定义一个所谓翻译起始符
        self.end_token = tf.placeholder(dtype=tf.int32,name='end-token-predict')   #翻译的结束符.

    def set_word_embedding_matrix(self):
        self.src_embedding_matrix=tf.Variable(initial_value=self.src_word_embeddings,trainable=True,name='src_embedding')
        self.dst_embedding_matrix=tf.Variable(initial_value=self.dst_word_embeddings,trainable=True,name='dst-embedding')

    def set_score_matrix(self):
        self.Ua=tf.get_variable(name='score-Ua',shape=[self.word_embedding_len,self.word_embedding_len],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.Wa=tf.get_variable(name='score-Wa',shape=[self.word_embedding_len,self.word_embedding_len],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.Va=tf.get_variable(name='score-Va',shape=[self.word_embedding_len,1],initializer=tf.contrib.layers.xavier_initializer())
    def build_graph(self):
        #encoder部分,可以使用现有的lstm结构
        with tf.name_scope("encoder"):
            self.src_data=tf.nn.embedding_lookup(self.src_embedding_matrix,ids=self.src_sentences)
            self.fw_lstm_encoder=tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums,activation=tf.nn.relu)
            self.bw_lstm_encoder=tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums,activation=tf.nn.relu)

            output,state=tf.nn.bidirectional_dynamic_rnn(self.fw_lstm_encoder,self.bw_lstm_encoder,inputs=self.src_data,sequence_length=self.src_real_sentence_lengths)

            self.fw_state=tf.concat((state[0]).c,(state[0]).h,-1)#batch_size,max_sequence,hidden_nums
            self.bw_state=tf.concat((state[1]).c,(state[1]).h,-1)#batch_size,max_sequence,hidden_nums

            self.contact=tf.concat([self.fw_state,self.bw_state],-1) #batch_size,max_sequence,2*hidden_nums
            shape_before_ = tf.shape(self.contact)

            self.contact=tf.reshape(self.contact,shape=[-1,4*self.lstm_hidden_nums])
            self.Wc_encoder=tf.get_variable(name='Wc-encoder',shape=[4*self.lstm_hidden_nums,self.lstm_hidden_nums],initializer=tf.contrib.layers.xavier_initializer)
            self.contact=tf.tanh(tf.matmul(self.contact,self.Wc_encoder))
            self.encoder_hiddens=tf.reshape(self.contact,shape=[shape_before_[0],shape_before_[1],self.lstm_hidden_nums],name='encoder-hidden-states')


        with tf.name_scope("decoder") :
            if self.train_flag == AttentionModel.TRAIN  :
                self.dst_data = tf.nn.embedding_lookup(ids=self.dst_sentences,params=self.dst_embedding_matrix)
                self.helper = TrainingHelper(inputs=self.dst_data,sequence_length=self.dst_real_sentence_lengths)

            elif self.train_flag ==AttentionModel.PREDICT   :
                #预测
                self.helper = GreedyEmbeddingHelper(self.dst_embedding_matrix,self.start_tokens,self.end_token)

            self.attention_mechanism = BahdanauAttention(num_units=self.lstm_hidden_nums,memory=self.encoder_hiddens,memory_sequence_length=self.src_real_sentence_lengths)
            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums)
            self.attn_cell = AttentionWrapper(self.decoder_cell,self.attention_mechanism,attention_layer_size=self.lstm_hidden_nums)
            self.decoder = BasicDecoder(self.attn_cell,self.helper,initial_state=self.attn_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size))
            final_outputs,final_states,final_sequence_length = dynamic_decode(self.decoder)
            if self.train_flag==AttentionModel.TRAIN:

                self.decoder_predict_logit = final_outputs.rnn_output

                self.targets = tf.reshape(self.dst_sentences,[-1])

                self.decoder_logit_flat=tf.reshape(self.decoder_predict_logit,shape=[-1,self.dst_vocb_size])
                self.cost = tf.losses.sparse_softmax_cross_entropy(labels=self.targets,self.decoder_logit_flat)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.train_op = optimizer.minimize(self.cost)
            elif self.train_flag==AttentionModel.PREDICT:
                self.decoder_predict_id = final_outputs.sample_id





