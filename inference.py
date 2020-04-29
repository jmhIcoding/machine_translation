__author__ = 'jmh081701'
import  tensorflow as tf
import  numpy as np
from  src.encoder_decoder.utils import  DATAPROCESS
batch_size = 100
dataGen = DATAPROCESS(batch_size=batch_size,
                        is_test= True,
                        seperate_rate=  0.0
                        )
            #所有的test里面的样本都拿去测试,seperate_rate 于是应该是100%,表示所有的样本都分离开来了

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph('checkpoints/dev.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')
    print("inference begin ")
    print("inference")

    dataGen.epoch = 1
    epoch =  dataGen.epoch
    while epoch == dataGen.epoch:
        output_x,output_label,src_sequence_length,dst_sequence_length=dataGen.next_train_batch()


        translate_logits=sess.run(fetches=logits,feed_dict={input_data:output_x,target_sequence_length:dst_sequence_length,source_sequence_length:src_sequence_length})

        for i in range(len(translate_logits)):
            obv=[dataGen.tokenid_to_size(output_x[i][j]) for j in range(len(output_x[i]))]

            label = [dataGen.tokenid_to_size(output_label[i][j]) for j in range(len(output_label[i]))]
            pre=[dataGen.tokenid_to_size(translate_logits[i][j]) * int(label[j]!=0) for j in range(len(translate_logits[i]))]
            print('src length:{0},dst length:{1}'.format(src_sequence_length[i],dst_sequence_length[i]))
            print({'obv':obv})
            print({'label':label})
            print({'prediction':pre})
            input('For next line, press enter.')
