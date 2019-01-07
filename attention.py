__author__ = 'jmh081701'
import  tensorflow as tf
import sys
import os
import tensorflow.contrib.crf
import copy
import time
import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(message)s")
logger = logging.getLogger(__name__)
sys.path.append(os.path.realpath("../"))
from utils import  DATAPROCESS
from tensorflow.contrib.seq2seq import *

dataGen = DATAPROCESS(source_ling_path="data/cn.txt",
                          dest_ling_path="data/en.txt",
                          source_word_embedings_path="data/cn.txt.ebd.npy",
                          source_vocb_path="data/cn.txt.vab",
                          dest_word_embeddings_path="data/en.txt.ebd.npy",
                          dest_vocb_path="data/en.txt.vab",
                          batch_size=100,
                          seperate_rate=0.1
                      )

class Model:
    def __init__(self):
        pass

class AttentionModel:
    TRAIN=1
    PREDICT=0
    def __init__(self,src_word_embeddings,dst_word_embeddings,
                 word_embedding_len,lstm_hidden_nums,src_max_sequence,dst_max_sequence,dst_vocb_size,train_flag,batch_size,
                 learning_rate,
                 start_token,
                 end_token,
                 model_path="paras/bilstm-models"
                ):
        '''
        :param src_word_embeddings:
        :param dst_word_embeddings:
        :param word_embedding_len:
        :param lstm_hidden_nums:
        :param src_max_sequence:
        :param dst_max_sequence:
        :param dst_vocb_size:
        :param train_flag:
        :param batch_size:
        :param learning_rate:
        :param start_token:     目标语料中，每个句子的起始<START>在目标语料词典中的id。注意目标语料的每个句子都得加上这种start
        :param end_token:       目标语料中，每个句子的结束符<END>在目标语料词典中的id。每个目标句子都得加上这种end标记
        :return:
        '''
        self.src_word_embeddings=src_word_embeddings
        self.dst_word_embeddings=dst_word_embeddings

        self.word_embedding_len=word_embedding_len
        self.src_sequence_max_length=src_max_sequence   #源语的最大句子长度
        self.dst_sequence_max_length=dst_max_sequence   #目标语的最大句子长度
        self.lstm_hidden_nums = lstm_hidden_nums


        self.dst_vocb_size=dst_vocb_size

        self.train_flag = train_flag
        self.batch_size = batch_size
        self.lr = learning_rate
        self.start_token_id=start_token
        self.end_token_id=end_token
        self.model_path = model_path
        self.set_placeholder()
        self.set_word_embedding_matrix()
        self.set_score_matrix()
        self.set_varibles()

        self.build_graph()
        self.sess = tf.Session()

        self.sess.run(tf.initialize_all_variables())
        self.Saver = tf.train.Saver()
        self.init_mode()
    def set_placeholder(self):
        self.src_sentences=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.src_sequence_max_length],name="input-src-sentences")#中文句子,形状应该是batch_size,句子最大填充长度
        self.dst_sentences=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.dst_sequence_max_length],name="input-dst-sentences")#英文句子,翻译的标准答案,batch_size,句子的最大填充长度
        self.src_real_sentence_lengths=tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='src-real-sequences-len')#中文句子的实际有效长度(去除填充的部分)
        self.dst_real_sentence_lengths=tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='dst-real-sequences-len')


    def set_varibles(self):
        self.start_tokens=tf.Variable([self.start_token_id]*self.batch_size,dtype=tf.int32,expected_shape=[self.batch_size],name='start-tokens',trainable=False)
        self.end_token = tf.Variable(initial_value=self.end_token_id,dtype=tf.int32,name='end-token',trainable=False)#翻译的结束符.
        #每个batch内的每个句子都要定义一个所谓翻译起始符

    def set_word_embedding_matrix(self):
        self.src_embedding_matrix=tf.Variable(initial_value=self.src_word_embeddings,trainable=True,name='src_embedding')
        self.dst_embedding_matrix=tf.Variable(initial_value=self.dst_word_embeddings,trainable=True,name='dst-embedding')

    def set_score_matrix(self):
        self.Ua=tf.get_variable(name='score-Ua',shape=[self.word_embedding_len,self.word_embedding_len],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.Wa=tf.get_variable(name='score-Wa',shape=[self.word_embedding_len,self.word_embedding_len],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        self.Va=tf.get_variable(name='score-Va',shape=[self.word_embedding_len,1],initializer=tf.contrib.layers.xavier_initializer())
    def build_graph(self):
        #encoder部分,可以使用现有的lstm结构
        with tf.variable_scope("encoder"):
            self.src_data=tf.nn.embedding_lookup(self.src_embedding_matrix,ids=self.src_sentences)
            self.fw_lstm_encoder=tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums,activation=tf.nn.relu)
            self.bw_lstm_encoder=tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums,activation=tf.nn.relu)

            outputs,state=tf.nn.bidirectional_dynamic_rnn(self.fw_lstm_encoder,self.bw_lstm_encoder,inputs=self.src_data,sequence_length=self.src_real_sentence_lengths,dtype=tf.float32)
            #自带的bidirectional_dynamic_rnn只输出最后一个隐藏层的状态,虽然他的output是没有问题的

            #print(output)
            self.fw_output=outputs[0]#batch_size,src_max_sequence,hidden_nums
            self.bw_output=outputs[1]#batch_size,src_max_sequence,hidden_nums
            #print(state)
            self.contact=tf.concat([self.fw_output,self.bw_output],-1) #batch_size,src_max_sequence,2*hidden_nums
            shape_before_ = tf.shape(self.contact)

            self.contact=tf.reshape(self.contact,shape=[-1,2*self.lstm_hidden_nums])
            #print(self.contact)
            self.Wc_encoder=tf.get_variable(name='Wc-encoder',shape=[2*self.lstm_hidden_nums,self.lstm_hidden_nums],initializer=tf.contrib.layers.xavier_initializer())
            self.contact=tf.tanh(tf.matmul(self.contact,self.Wc_encoder))#乘完以后又映射回来了
            #print(self.contact)
            self.encoder_hiddens=tf.reshape(self.contact,shape=[shape_before_[0],shape_before_[1],self.lstm_hidden_nums],name='encoder-hidden-states')

            self.encdoer_final_state = state[0].h+state[1].c
        with tf.name_scope("decoder") :
            self.dst_data = tf.nn.embedding_lookup(ids=self.dst_sentences,params=self.dst_embedding_matrix)
            self.train_helper = TrainingHelper(inputs=self.dst_data,sequence_length=self.dst_real_sentence_lengths)
            self.infe_helper = GreedyEmbeddingHelper(self.dst_embedding_matrix,self.start_tokens,self.end_token)

            self.attention_mechanism = BahdanauAttention(num_units=self.lstm_hidden_nums,memory=self.encoder_hiddens,memory_sequence_length=self.src_real_sentence_lengths)
            self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_hidden_nums)
            self.attn_cell = AttentionWrapper(self.decoder_cell,self.attention_mechanism,attention_layer_size=self.lstm_hidden_nums)
            self.decoder_output_layer = tf.layers.Dense(units=self.dst_vocb_size,activation=tf.nn.softmax,use_bias=False)

            with tf.variable_scope("decoder"):
                print("Train Section")
                self.train_decoder = BasicDecoder(self.attn_cell,self.train_helper,initial_state=self.attn_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size),output_layer=self.decoder_output_layer)
                final_outputs,final_states,final_sequence_length = dynamic_decode(self.train_decoder,maximum_iterations=self.dst_sequence_max_length)
                self.decoder_train_logit = final_outputs.rnn_output
                self.decoder_train_id = final_outputs.sample_id

            with tf.variable_scope('decoder',reuse=True):
                print("ATTENTIONMODEL PREDICT Section")
                self.inference_decoder=BasicDecoder(self.attn_cell,self.infe_helper,initial_state=self.attn_cell.zero_state(dtype=tf.float32,batch_size=self.batch_size),output_layer=self.decoder_output_layer)
                inf_outputs,_,_=dynamic_decode(self.inference_decoder,maximum_iterations=self.dst_sequence_max_length)
                self.decoder_inference_logit =inf_outputs.rnn_output
                self.decoder_inference_id = inf_outputs.sample_id

            with tf.variable_scope('train-section'):

                self.targets = tf.reshape(self.dst_sentences,[self.dst_sequence_max_length*self.batch_size],name='input-target-id')  #给定的监督语料
                print(self.decoder_train_logit)
                self.decoder_logit_flat=tf.reshape(self.decoder_train_logit,shape=[-1,self.dst_vocb_size],name='decoder-logit-flat')
                print(self.batch_size,self.dst_sequence_max_length,self.dst_vocb_size)
                self.mask = tf.to_float(tf.sequence_mask(self.dst_real_sentence_lengths,maxlen=self.dst_sequence_max_length),'mask')
                self.seq_loss = sequence_loss(self.decoder_train_logit,self.dst_sentences,weights=self.mask)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                #self.train_op = self.optimizer.minimize(self.seq_loss)
                self.params = tf.trainable_variables()
                self.gradients = tf.gradients(self.seq_loss,self.params)
                #print("vars for loss function: ", self.vars)
                self.clipped_gradients= [(tf.clip_by_va(self.grad,-5,5),var) for grad,var in self.gradients if self.gradients if not None ] # clip gradients
                self.train_op = self.optimizer.apply_gradients(self.clipped_gradients)

    def save(self):
        #保存模型参数
        logger.info("#"*30)
        logger.info("............SAVE MODEL..........")
        self.Saver.save(self.sess,self.model_path)
    def load(self):
        #载入模型参数
        if os.path.exists(self.model_path+".meta") and os.path.exists(self.model_path+".index"):
            logger.info(".............LOAD MODEL..........")
            self.Saver.restore(self.sess,self.model_path)
            logging.info('load a historical model.')
        else:
            logger.warning("LOAD MODEL ERROR")
            logging.info('re train a new  model.')
    def init_mode(self):
        #初始化参数
        self.sess.run(tf.initialize_all_variables())
        #载入已有模型
        self.load()

    def one_step(self,one_batch_data):
        if self.train_flag ==AttentionModel.TRAIN:
            fetches=[self.train_op,self.seq_loss,self.decoder_train_id]
            feed_dict=\
                {
                self.src_sentences:one_batch_data[0],
                self.dst_sentences:one_batch_data[1],
                self.src_real_sentence_lengths:one_batch_data[2],
                self.dst_real_sentence_lengths:one_batch_data[3]
            }
        elif self.train_flag==AttentionModel.PREDICT:
            # src_sentence,[],[],[]
            print("one step")
            fetches=[self.decoder_inference_id]
            feed_dict={
                self.src_sentences:one_batch_data[0],
                self.src_real_sentence_lengths:one_batch_data[2]
            }
        else:
            logger.log(level=logging.ERROR,msg="unknow mode.")
            return  None
        results=self.sess.run(fetches,feed_dict)
        return results




