__author__ = 'jmh081701'
from attention import  AttentionModel
from attention import  logger
from utils import  DATAPROCESS
from utils import  evaluate
hidden_nums=600
learning_rate = 0.0001
MODE = AttentionModel.PREDICT
max_epoch = 30
dataGen = DATAPROCESS(
                        source_ling_path="data/cn.txt",
                          dest_ling_path="data/en.txt",
                          source_word_embedings_path="data/cn.txt.ebd.npy",
                          source_vocb_path="data/cn.txt.vab",
                          dest_word_embeddings_path="data/en.txt.ebd.npy",
                          dest_vocb_path="data/en.txt.vab",
                          batch_size=200,
                          seperate_rate=0.1
                        )

Model = AttentionModel(src_word_embeddings=dataGen.src_word_embeddings,dst_word_embeddings=dataGen.dst_word_embeddings,word_embedding_len=dataGen.embedding_length
                       ,lstm_hidden_nums=hidden_nums,
                       src_max_sequence=dataGen.src_sentence_length,dst_max_sequence=dataGen.dst_sentence_length,
                       dst_vocb_size=dataGen.dst_vocb_size,
                       train_flag=MODE,
                       batch_size=dataGen.batch_size,
                       learning_rate=learning_rate,
                       start_token=int(dataGen.dst_word2id['<START>']),
                       end_token=int(dataGen.dst_word2id['<END>']),
                       model_path=".\\bilstm-models.cpk"
                       )


while dataGen.epoch < max_epoch:
        if MODE ==AttentionModel.TRAIN:
            output_x,output_label,src_sequence_length,dst_sequence_length=dataGen.next_train_batch()
            _,loss,predict_id =Model.one_step([output_x,output_label,src_sequence_length,dst_sequence_length])
            logger.info({'loss':loss,'epoch':dataGen.epoch,'batch':dataGen.train_batch_index})
        elif MODE==AttentionModel.PREDICT:
            print("inference begin ")
            output_x,output_label,src_sequence_length,dst_sequence_length=dataGen.test_data()
            predict_id=Model.one_step(one_batch_data=[output_x,output_label,src_sequence_length,dst_sequence_length])
            print(predict_id[:10])
            print("inference")
            break
        if dataGen.train_batch_index is 0:
            Model.save()





