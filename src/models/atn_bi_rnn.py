'''
Created on 2017年3月23日

@author: zry
'''
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn

class Atn_bi_rnn:
    def __init__(self, data_dir=None):
        # Parameters
        self.learning_rate = 0.0005
        self.batch_size = 300
        # Network Parameter
        self.max_sentence_len = 80
        self.word_size = 50 #词向量维度
        self.position_size = int(15*2) #与命名体相对位置的维度
        self.pos_size = 0 #词性维度
        self.input_size = int(self.word_size+self.position_size+self.pos_size)
        self.bi_rnn_hidden = int(150*2)
        self.attention_hidden = 80
        self.relation_classes = 10
        
        
    def inference(self):
        weights = {
            'attention': tf.Variable(tf.random_normal([int(self.bi_rnn_hidden), self.bi_rnn_hidden]), name='attention_w'),
            'attention_h': tf.Variable(tf.random_normal([self.bi_rnn_hidden, 1]), name='attention_h_w'),
            'attention_r': tf.Variable(tf.random_normal([self.bi_rnn_hidden*2, 1]), name='attention_r_w'),
            'attention_out': tf.Variable(tf.random_normal([self.bi_rnn_hidden, self.attention_hidden]), name='attention_out'),
            'out': tf.Variable(tf.random_normal([self.attention_hidden, self.relation_classes]), name='out_w')
        }
        biases = {
            'attention': tf.Variable(tf.random_normal([self.max_sentence_len]), name='attention'),
            'out': tf.Variable(tf.random_normal([self.relation_classes]), name='out_b')
        }
        #-------------------------------------- inputs ----------------------------------------------
        sentence_input = tf.placeholder('float', [None, self.max_sentence_len, self.input_size], name='sentence_input')
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
#             lstm_cell_mix = rnn.BasicLSTMCell(self.bi_rnn_hidden)
            bi_rnn_outs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, sentence_input, sequence_length= sentence_len_input, dtype=tf.float32)
            bi_rnn_out,_ = tf.nn.dynamic_rnn(lstm_cell_mix, tf.concat(bi_rnn_outs, -1), sequence_length= sentence_len_input, dtype=tf.float32)
            bi_rnn_out_last = bi_rnn_out[:,-1,:]
        #-------------------------------------- attention_layer ----------------------------------------------
        with tf.variable_scope('attention_layer'):
            # 按entity位置在rnn_out中查询，找到entity对应的rnn_out
            entity_rnn_out = tf.matmul(entity_input_onehot,bi_rnn_out)
            entity_rnn_out = tf.concat([entity_rnn_out[:,0,:],entity_rnn_out[:,1,:]],1)
#             entity_rnn_out = tf.nn.dropout(entity_rnn_out, dropout_input[1])
            # attention rate:softmax(rnn_last*W*rnn_out)
#             attention_left = tf.expand_dims(tf.matmul(bi_rnn_out_last, weights['attention']), 1)
#             attention_rate = tf.nn.softmax(tf.matmul(attention_left, tf.transpose(bi_rnn_out,[0,2,1])))
#             attention_out = tf.reshape(tf.matmul(attention_rate, bi_rnn_out), [-1,self.bi_rnn_hidden])
#             attention_out = tf.matmul(attention_out, weights['attention_out'])
            # attention rate:softmax(tanh(rnn_out*W1 + rnn_Last*W2 + b))
            attention_r = tf.matmul(entity_rnn_out, weights['attention_r'])
            attention_r = tf.matmul(attention_r, tf.zeros([1,self.max_sentence_len])+1) #复制多份，适应句长
            attention_h = tf.reshape(tf.matmul(tf.reshape(bi_rnn_out,[-1,self.bi_rnn_hidden]), weights['attention_h']), [-1,self.max_sentence_len])
            attention_rate = tf.expand_dims(tf.nn.softmax(tf.nn.tanh(attention_h + attention_r + biases['attention'])) ,1)
            # attention rate:softmax(V*(rnn_out*rnn_Last))
#             batch_size = tf.shape(bi_rnn_out_last)[0]
#             bi_rnn_out_last = tf.expand_dims(bi_rnn_out_last,1)
#             attention_r = tf.matmul(tf.zeros([batch_size,self.max_sentence_len,1])+1, bi_rnn_out_last) #复制多份，适应句长
#             attention_mul = tf.reshape(tf.multiply(attention_r,bi_rnn_out),[-1,self.bi_rnn_hidden])
#             attention_rate = tf.reshape(tf.nn.softmax(tf.matmul(attention_mul, weights['attention_h'])) ,[-1,1,self.max_sentence_len])
            attention_out = tf.matmul(tf.reshape(tf.matmul(attention_rate, bi_rnn_out), [-1,self.bi_rnn_hidden]), weights['attention_out'])
        with tf.variable_scope('output'):
            attention_out_dropout = tf.nn.dropout(attention_out, dropout_input[1])
            pred_label = tf.matmul(tf.nn.sigmoid(attention_out_dropout), weights['out']) + biases['out']
        return pred_label, sentence_input, sentence_len_input, entity_input, dropout_input
    
    
    def loss_evaluation(self, pred_label):
        #-------------------------------------- Define loss and evaluation --------------------------------------
        correct_label_input = tf.placeholder('int32', [None, self.relation_classes], name='correct_label_input')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=correct_label_input))
    #     correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(label,1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return loss, correct_label_input
    
    
    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
#         optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        return optimizer, init
    
