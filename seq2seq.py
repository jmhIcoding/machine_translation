#coding:utf8
__author__ = 'jmh081701'
from src.encoder_decoder.utils import  DATAPROCESS
import  tensorflow as tf
import datetime
from tensorflow.python.ops import array_ops
batch_size = 100
rnn_size = 200
rnn_num_layers = 1

max_epoch = 500


encoder_embedding_size = 50
decoder_embedding_size = 50
# Learning Rate
lr = 1e-3
factor = 0.98  #学习速率的衰减因子
display_step = 10
dataGen = DATAPROCESS(batch_size=batch_size,
                        seperate_rate=0
                        )

def model_inputs():
    inputs = tf.placeholder(tf.int32, [batch_size, dataGen.src_sentence_length], name="inputs")
    targets = tf.placeholder(tf.int32, [batch_size, dataGen.dst_sentence_length], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    source_sequence_len = tf.placeholder(tf.int32, (batch_size,), name="source_sequence_len")
    target_sequence_len = tf.placeholder(tf.int32, (batch_size,), name="target_sequence_len")
    max_target_sequence_len = tf.placeholder(tf.int32, (batch_size,), name="max_target_sequence_len")

    return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len

def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
                  source_sequence_len, source_vocab_size, encoder_embedding_size=100):

    # 对输入的单词进行词向量嵌入
    encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)
    #print(encoder_embed.shape)
    # LSTM单元
    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        return lstm

    # 堆叠rnn_num_layers层LSTM
    lstms = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstms, encoder_embed,
                                                        sequence_length=source_sequence_len,
                                                        dtype=tf.float32)
    #print(encoder_outputs.shape,encoder_states.shape)
    return encoder_outputs, encoder_states

def decoder_layer_inputs(target_data,  batch_size):
    """
    对Decoder端的输入进行处理

    @param target_data: 目标语数据的tensor
    @param target_vocab_to_int: 目标语数据的词典到索引的映射: dict
    @param batch_size: batch size
    """
    # 去掉batch中每个序列句子的最后一个单词
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    decoder_inputs = tf.concat([tf.fill([batch_size, 1], dataGen.start_token_id),ending], 1)

    return decoder_inputs


def decoder_layer(encoder_states, decoder_inputs, target_sequence_len,
                   max_target_sequence_len, rnn_size, rnn_num_layers,
                   target_vocab_size, decoder_embedding_size, batch_size,
                   start_id,end_id,encoder_outputs):

    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
        return lstm

    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])


    output_layer = tf.layers.Dense(target_vocab_size)

    with tf.variable_scope("decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                           sequence_length=target_sequence_len,
                                                           time_major=False)
        print('88,',decoder_embed_input.shape)
        #添加注意力机制
        #先定义一个Bahda 注意力机制。它是用一个小的神经网络来做打分的,num_units指明这个小的神经网络的大小
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,memory=encoder_outputs,memory_sequence_length=source_sequence_len)
        #在原来rnn的基础上配上一层AttentionWrapper
        decoder_cell_train = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,attention_mechanism,attention_layer_size=rnn_size)
        #初始状态设置为encoder最后的输出状态
        #初始状态设置为encoder最后的输出状态
        training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell_train,
                                                          training_helper,
                                                          decoder_cell_train.zero_state(batch_size,dtype=tf.float32).clone(cell_state=encoder_states),
                                                          output_layer)
        print('good94,max_target_sequence_len',max_target_sequence_len)
        training_decoder_outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                      impute_finished=False,
                                                                      maximum_iterations=max_target_sequence_len)
        print('98:',training_decoder_outputs.rnn_output.shape)
        print('good98')
    with tf.variable_scope("decoder", reuse=True):

        start_tokens = tf.tile(tf.constant([dataGen.start_token_id], dtype=tf.int32), [batch_size], name="start_tokens")
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                    start_tokens,
                                                                    dataGen.end_token_id)
        print('good105')
        attention_mechanism_infer = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,memory=encoder_outputs,memory_sequence_length=source_sequence_len)
            #加入attention ，加入的方式与train类似
        decoder_cell_infer = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,attention_mechanism_infer,attention_layer_size=rnn_size)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell_infer,
                                                           inference_helper,
                                                           decoder_cell_infer.zero_state(batch_size,dtype=tf.float32).clone(cell_state=encoder_states),
                                                           output_layer)
        print('good110')
        inference_decoder_outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                          impute_finished=False,
                                                                          maximum_iterations=max_target_sequence_len)
        print('good114')

    return training_decoder_outputs, inference_decoder_outputs



def seq2seq_model(input_data, target_data, batch_size,
                 source_sequence_len, target_sequence_len, max_target_sentence_len,
                 source_vocab_size, target_vocab_size,
                 encoder_embedding_size, decoder_embeding_size,
                 rnn_size, rnn_num_layers):
    encoder_outputs, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
                                      source_vocab_size, encoder_embedding_size)

    decoder_inputs = decoder_layer_inputs(target_data, batch_size)

    training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_states=encoder_states,
                                                                       decoder_inputs=decoder_inputs,
                                                                      target_sequence_len=target_sequence_len,
                                                                       max_target_sequence_len=max_target_sentence_len,
                                                                      rnn_size=rnn_size,
                                                                      rnn_num_layers=rnn_num_layers,
                                                                      target_vocab_size=target_vocab_size,
                                                                      decoder_embedding_size=decoder_embeding_size,
                                                                       batch_size=batch_size,
                                                                       start_id=dataGen.start_token_id,
                                                                       end_id=dataGen.end_token_id,
                                                                       encoder_outputs=encoder_outputs)
    print('good139')
    return training_decoder_outputs, inference_decoder_outputs


train_graph = tf.Graph()

with train_graph.as_default():
    inputs, targets, learning_rate, source_sequence_len, target_sequence_len, _ = model_inputs()

    max_target_sequence_len = dataGen.dst_sentence_length
    train_logits, inference_logits = seq2seq_model(
                                                   input_data= inputs,
                                                  target_data= targets,
                                                  batch_size=batch_size,
                                                  source_sequence_len=source_sequence_len,
                                                  target_sequence_len=target_sequence_len,
                                                  max_target_sentence_len=max_target_sequence_len,
                                                  source_vocab_size=dataGen.vocb_size,
                                                  target_vocab_size= dataGen.vocb_size,
                                                  encoder_embedding_size= encoder_embedding_size,
                                                  decoder_embeding_size= decoder_embedding_size,
                                                  rnn_size=rnn_size,
                                                  rnn_num_layers=rnn_num_layers)

    training_logits = tf.identity(train_logits.rnn_output, name="logits")
    inference_logits = tf.identity(inference_logits.sample_id, name="predictions")

    print('170:',target_sequence_len.get_shape())
    with tf.name_scope("optimization"):
        current_ts = tf.to_int32(tf.minimum(tf.shape(targets)[1], tf.shape(training_logits)[1]))
        # 对 target 进行截取
        target_sequences = tf.slice(targets, begin=[0, 0], size=[-1, current_ts])
        masks = tf.sequence_mask(lengths=target_sequence_len, maxlen=current_ts, dtype=tf.float32)
        training_logits = tf.slice(training_logits, begin=[0, 0, 0], size=[-1, current_ts, -1])

        inference_sample_id = tf.slice(inference_logits, begin=[0, 0], size=[-1, current_ts])
        #valid_loss = tf.reduce_mean(tf.equal(target_sequences,inference_sample_id))
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                target_sequences,
                                                masks)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(clipped_gradients)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    try:
        loader = tf.train.Saver()
        loader.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
        print('Load from a history model')
    except Exception as exp:
        print("Train a new model")
    saver = tf.train.Saver()
    dataGen.epoch =1
    step = 1
    epoch  = dataGen.epoch
    while dataGen.epoch < max_epoch:
            output_x,output_label,src_sequence_length,dst_sequence_length=dataGen.next_train_batch()
            _, loss = sess.run(
                [train_op, cost],
                {inputs: output_x,
                 targets: output_label,
                 learning_rate: lr,
                 source_sequence_len: src_sequence_length,
                 target_sequence_len: dst_sequence_length})
            if dataGen.train_batch_index % display_step == 0 and dataGen.train_batch_index > 0:
                print('{0} : Epoch {1:>3} Step {3} - Loss On 验证集: {2:>6.4f}'
                      .format(datetime.datetime.now(),dataGen.epoch, loss,step))

            step += 1
            if epoch != dataGen.epoch :
                epoch = dataGen.epoch
                lr *=factor
                saver.save(sess,"checkpoints/dev")
                #每次epoch保存一次模型
    # Save Model
    saver.save(sess, "checkpoints/dev")
    print('Model Trained and Saved')
