'''
Created on 2017年3月23日

@author: zry
'''
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn

class Rntn_bi_rnn:
    def __init__(self, data_dir=None):
        # Parameters
        self.learning_rate = 0.001
        self.batch_size = 300
        # Network Parameter
        self.max_sentence_len = 80
        self.word_size = 50 #词向量维度
        self.position_size = int(15*2) #与命名体相对位置的维度
        self.pos_size = 0 #词性维度
        self.input_size = int(self.word_size+self.position_size)
        self.bi_rnn_hidden = int(80*2)
        self.rntn_hidden = 80
        self.relation_classes = 10
        
        
    def inference(self):
        weights = {
            'rntn': tf.Variable(tf.random_normal([self.bi_rnn_hidden*2, self.bi_rnn_hidden*2*self.rntn_hidden]), name='rntn_w'),
            'rntn_bias': tf.Variable(tf.random_normal([self.bi_rnn_hidden*2, self.rntn_hidden]), name='rntn_bias_w'),
            'out': tf.Variable(tf.random_normal([self.rntn_hidden, self.relation_classes]), name='out_w')
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.relation_classes]), name='out_b')
        }
        #-------------------------------------- inputs ----------------------------------------------
        word_list_input = tf.placeholder('float', [None, self.max_sentence_len, self.input_size], name='word_list_input')
        sentence_len_input = tf.placeholder('int32', [None], name='sentence_len_input')
        dropout_input = tf.placeholder('float', [2], name='dropout_input')
        # 按entity位置在rnn_out中查询，找到entity对应的rnn_out
        entity_input = tf.placeholder('int32', [None,2], 'entity_input')
        entity_input_onehot = tf.one_hot(entity_input, self.max_sentence_len)
        #-------------------------------------- bi-rnn ----------------------------------------------
        with tf.variable_scope('bi-rnn') as scope:
            lstm_cell_fw = rnn.DropoutWrapper(rnn.BasicLSTMCell(int(self.bi_rnn_hidden/2)), output_keep_prob=dropout_input[0])
            lstm_cell_bw = rnn.DropoutWrapper(rnn.BasicLSTMCell(int(self.bi_rnn_hidden/2)), output_keep_prob=dropout_input[0])
            lstm_cell_mix = rnn.BasicLSTMCell(self.bi_rnn_hidden)
            bi_rnn_outs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, word_list_input, sequence_length= sentence_len_input, dtype=tf.float32)
            bi_rnn_out,_ = tf.nn.dynamic_rnn(lstm_cell_mix, tf.concat(bi_rnn_outs, -1), sequence_length= sentence_len_input, dtype=tf.float32)
            bi_rnn_out_last = bi_rnn_out[:,-1,:]
        #-------------------------------------- attention_layer ----------------------------------------------
        with tf.variable_scope('rntn_layer'):
            # 按entity位置在rnn_out中查询，找到entity对应的rnn_out
            entity_rnn_out = tf.matmul(entity_input_onehot,bi_rnn_out)
            entity_rnn_out = tf.concat([entity_rnn_out[:,0,:],entity_rnn_out[:,1,:]],1)
            # rntn layer:[e1,e2]*V*[e1,e2]+W*[e1,e2],其中V为三维权值矩阵
            rntn_power_left = tf.reshape(tf.matmul(entity_rnn_out,weights['rntn']), [-1,self.rntn_hidden,self.bi_rnn_hidden*2])
            rntn_power = tf.reshape(tf.matmul(rntn_power_left, tf.expand_dims(entity_rnn_out,2)), [-1,self.rntn_hidden])
            rntn_out = rntn_power + tf.matmul(entity_rnn_out, weights['rntn_bias'])
        with tf.variable_scope('output'):
            out_dropout = tf.nn.dropout(rntn_out, dropout_input[1])
            pred_label = tf.matmul(tf.nn.sigmoid(out_dropout), weights['out']) + biases['out']
        return pred_label, word_list_input, sentence_len_input, entity_input, dropout_input
    
    
    def loss_evaluation(self, pred_label):
        #-------------------------------------- Define loss and evaluation --------------------------------------
        correct_label_input = tf.placeholder('int32', [None, self.relation_classes], name='correct_label_input')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=correct_label_input))
    #     correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(label,1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return loss, correct_label_input
    
    
    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        return optimizer, init
    
