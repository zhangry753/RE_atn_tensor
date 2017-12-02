'''
Created on 2017年4月17日

@author: zry
'''
import numpy
from numpy.core import numeric

from models.atn_cnn import Atn_cnn
import tensorflow as tf
from util.batch_reader_cnn import Batch_reader_cnn


def display_data(model, batch_reader, sess, use_test_data=True, output_file=None):
    if use_test_data:
        word_vecs_batch,entity_index_batch, position_vecs_batch, labels_batch = batch_reader.read_batch_test(model.max_sentence_len,500)
    else:
        word_vecs_batch, entity_index_batch, position_vecs_batch, labels_batch = batch_reader.read_batch(500, model.max_sentence_len)
    feed_dict = {
        words_input:word_vecs_batch, 
        entity_input:entity_index_batch,
        position_input:position_vecs_batch, 
        correct_label_input:labels_batch,
        dropout_input:[1,1]
    }
    pred_y = sess.run(tf.argmax(y,-1), feed_dict=feed_dict)
    correct_y = sess.run(tf.argmax(labels_batch,-1), feed_dict=feed_dict)
    acc_map = {i:[0]*2 for i in range(10)}
    for i in range(len(pred_y)):
        if pred_y[i] == correct_y[i]:
            acc_map[correct_y[i]][0] += 1
        else:
            acc_map[correct_y[i]][1] += 1
    print(acc_map)
    correct_count = 0
    for _,item in acc_map.items():
        correct_count += item[0]
    print(correct_count-acc_map[0][0])
    if output_file != None:
        for relation,acc_list in acc_map.items():
            output_file.write(str(relation)+':'+str(acc_list[0])+','+str(acc_list[1])+'\t')
        output_file.write('\n'+str(correct_count-acc_map[0][0])+'\n')
    return correct_count-acc_map[0][0]
        
        
def output_test_pred_label(model, batch_reader, sess):
    word_vecs_batch,entity_index_batch, position_vecs_batch, labels_batch = batch_reader.read_batch_test(model.max_sentence_len)
    feed_dict = {
        words_input:word_vecs_batch,
        position_input:position_vecs_batch,
        entity_input:entity_index_batch, 
        correct_label_input:labels_batch,
        dropout_input:[1,1]
    }
    rel_class = [
            'Other',
            'Cause-Effect',
        'Product-Producer',
           'Entity-Origin',
       'Instrument-Agency',
         'Component-Whole',
       'Content-Container',
      'Entity-Destination',
       'Member-Collection',
           'Message-Topic'
       ]
    pred_y = sess.run(tf.argmax(y,-1), feed_dict=feed_dict)
    output_file = open(r'D:\myDoc\study\语料\SemEval2010_task8\SemEval2010_task8_test\pred.txt','w')
    for i,label_index in enumerate(pred_y):
        if label_index == 0:
            output_file.write(str(i+1) + '\t' + rel_class[label_index]+'\n')
        else:
            output_file.write(str(i+1) + '\t' + rel_class[label_index]+'(e1,e2)\n')
    output_file.close()
    

if __name__ == '__main__':
    output_path = r'C:\Users\zry\Desktop\test.txt'
    checkpoint_dir = 'C:/Users/zry/Desktop/checkpoint/'
    output_file = open(output_path,'w')
    
    model = Atn_cnn()
    batch_reader = Batch_reader_cnn()
    y,words_input,position_input,entity_input,dropout_input = model.inference()
    loss,correct_label_input = model.loss_evaluation(y)
    optimizer,init = model.train(loss)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        tf.summary.scalar('loss', loss)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(checkpoint_dir+"logs")
#         saver.restore(sess, r'C:\Users\zry\Desktop\checkpoint\model.ckpt-2200')
#         output_test_pred_label(model, batch_reader, sess)
        sess.run(init)
        #---------------------------------- train part ---------------------------------------------------------------
        for step in range(50001):
            word_vecs_batch, entity_index_batch, position_vecs_batch, labels_batch = batch_reader.read_batch(model.batch_size, model.max_sentence_len)
            feed_dict = {
                words_input:word_vecs_batch, 
                entity_input:entity_index_batch,
                position_input:position_vecs_batch, 
                correct_label_input:labels_batch,
                dropout_input:[1,1]
            }
            sess.run(optimizer, feed_dict=feed_dict)
            if step%5==0:
                summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if step%200==0 and step>=3000:
                # display trial and test data
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)
                print("--------------"+str(step)+"-------------")
                output_file.write("--------------"+str(step)+"-------------\n")
                display_data(model, batch_reader, sess, False,output_file)
                test_correct_count = display_data(model, batch_reader, sess,output_file=output_file)
                print()
                output_file.write('\n')
                output_file.flush()
