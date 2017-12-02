'''
Created on 2017年4月17日

@author: zry
'''
import random
import tensorflow as tf
from models.atn_bi_rnn import Atn_bi_rnn
from models.rntn_bi_rnn import Rntn_bi_rnn
from models.rntn_atn_bi_rnn import Rntn_atn_bi_rnn
from models.atn_rntn_bi_rnn import Atn_rntn_bi_rnn
from models.atn_cnn import Atn_cnn
from util.batch_reader import Batch_reader
import os
from models import rntn_atn_bi_rnn

def get_F1(y_pred, y_label):
  y_pred = tf.one_hot(tf.argmax(y_pred, -1), 10)
  y_label = tf.cast(y_label, tf.float32)
  y_pred_rel = y_pred[:,1:]
  y_label_rel = y_label[:,1:]
  correct_rel = tf.reduce_sum(y_pred_rel * y_label_rel)
  total_correct_rel = tf.reduce_sum(y_label_rel)
  total_pred_rel = tf.reduce_sum(y_pred_rel)
  acc = correct_rel / total_pred_rel
  recall = correct_rel / total_correct_rel
  F1 = 2*acc*recall / (acc+recall)
  return F1

def display_data(model, batch_reader, sess, use_test_data=True):
    if use_test_data:
        senten_vecs_batch, sentence_len_batch, entity_indexs_batch, labels_batch = batch_reader.read_batch_test(1500)
    else:
        senten_vecs_batch, sentence_len_batch, entity_indexs_batch, labels_batch = batch_reader.read_batch(500)
    feed_dict = {
      sentence_input:senten_vecs_batch, 
      sentence_len_input:sentence_len_batch,
      entity_input:entity_indexs_batch, 
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
    relation_count = 0
    for i in range(1,len(acc_map)):
        correct_count += acc_map[i][0]
        relation_count += acc_map[i][0]+acc_map[i][1]
    if acc_map[0][0]==0 or relation_count==0 or correct_count==0:
        acc_F1 = 0
        acc_pos = 0
        acc_neg = 0
    else:
        acc_pos = correct_count/relation_count
#         acc_neg = acc_map[0][0]/(acc_map[0][0]+acc_map[0][1])
        acc_neg = correct_count/(correct_count+acc_map[0][1])
        acc_F1 = 2*acc_pos*acc_neg / (acc_pos+acc_neg)
    print("%.4f\t%.4f\t%.4f"%(acc_pos,acc_neg,acc_F1))
    
    
if __name__ == '__main__':
  checkpoint_dir = 'C:/Users/zry/Desktop/checkpoint/'
  
  model = Rntn_atn_bi_rnn()
  batch_reader = Batch_reader(model.word_size,model.position_size,model.pos_size,model.max_sentence_len)
  y,sentence_input,sentence_len_input,entity_input,dropout_input = model.inference()
  loss,correct_label_input = model.loss_evaluation(y)
  optimizer,init = model.train(loss)
  F1_score = get_F1(y, correct_label_input)
  senten_vecs_batch, sentence_len_batch, entity_indexs_batch, labels_batch = batch_reader.read_batch_test(1000)
  feed_dict_test = {
    sentence_input:senten_vecs_batch, 
    sentence_len_input:sentence_len_batch,
    entity_input:entity_indexs_batch, 
    correct_label_input:labels_batch,
    dropout_input:[1,1]
  }
  with tf.Session() as sess:
    saver = tf.train.Saver(max_to_keep=50)
    F1_score_summary = tf.placeholder(tf.float32, [])
    tf.summary.scalar('F1_original', F1_score_summary)
    summary_writer = tf.summary.FileWriter(checkpoint_dir+"logs")
#     saver.restore(sess, r'C:\Users\zry\Desktop\checkpoint\model.ckpt-800')
#     output_test_pred_label(model, batch_reader, sess)
#     display_data(model, batch_reader, sess)
    sess.run(tf.global_variables_initializer())
    #---------------------------------- train part ---------------------------------------------------------------
    F1_sum = 0.
    for step in range(0,1501):
      senten_vecs_batch, sentence_len_batch, entity_indexs_batch, labels_batch = batch_reader.read_batch(model.batch_size)
      feed_dict = {
        sentence_input:senten_vecs_batch, 
        sentence_len_input:sentence_len_batch,
        entity_input:entity_indexs_batch, 
        correct_label_input:labels_batch,
        dropout_input:[0.5,0.5]
      }
      sess.run(optimizer, feed_dict=feed_dict)
      F1_step = sess.run(F1_score, feed_dict=feed_dict_test)
      F1_sum += F1_step
      
      if step%50==0 and step>0:
        F1 = F1_sum/50
        summary_str = sess.run(tf.summary.merge_all(), 
                               feed_dict={F1_score_summary:F1})
        summary_writer.add_summary(summary_str, step/5)
        F1_sum = 0.
        display_data(model, batch_reader, sess)
        
#       if step%200==0:
#         saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=step)
