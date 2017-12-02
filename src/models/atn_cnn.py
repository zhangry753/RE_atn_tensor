'''
Created on 2017年3月23日

@author: zry
'''
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn

class Atn_cnn:
    def __init__(self, data_dir=None):
        # Parameters
        self.learning_rate = 0.001
        self.batch_size = 200
        # Network Parameter
        self.max_sentence_len = 80
        self.word_size = 50 #词向量维度
        self.position_size = int(15*2) #与命名体的相对位置维度
        self.input_size = int(self.word_size+self.position_size)
        self.conv_window_size = 3
        self.conv_size = 200
        self.full_conn_size = 100
        self.relation_classes = 10
        
        
    def inference(self):
        weights = {
            'input_attention': tf.get_variable("input_attention_w", [self.input_size, self.input_size], initializer=tf.random_normal_initializer()),
#             'pool_attention': tf.get_variable("pool_attention_w", [self.full_conn_size, self.pool_out_size], initializer=tf.random_normal_initializer()),
            'conv': tf.Variable(tf.random_normal([self.conv_window_size, self.input_size, self.conv_size]), name='conv_w'),
            'full_conn': tf.Variable(tf.random_normal([self.conv_size, self.full_conn_size]), name='full_conn_w'),
            'out': tf.Variable(tf.random_normal([self.full_conn_size, self.relation_classes]), name='out_w')
        }
        biases = {
            'conv': tf.Variable(tf.random_normal([self.conv_size]), name='conv_b'),
            'full_conn': tf.Variable(tf.random_normal([self.full_conn_size]), name='full_conn_b'),
            'out': tf.Variable(tf.random_normal([self.relation_classes]), name='out_b')
        }
        #-------------------------------------- input ----------------------------------------------
        dropout_input = tf.placeholder('float', [2], name='dropout_input')
        words_input = tf.placeholder('float', [None, self.max_sentence_len, self.word_size], name='words_input')
        position_input = tf.placeholder('float', [None, self.max_sentence_len, self.position_size], name='position_input')
        input_vecs = tf.concat([words_input,position_input],-1)
        input_dropout = tf.nn.dropout(input_vecs, dropout_input[0])
        entity_input = tf.placeholder('int32', [None,2], 'entity_input')
        entity_input_onehot = tf.one_hot(entity_input, self.max_sentence_len)
        entity_input_vecs = tf.matmul(entity_input_onehot,input_vecs)
        e1_input = entity_input_vecs[:,0,:]
        e2_input = entity_input_vecs[:,1,:]
        #-------------------------------------- input attention ----------------------------------------------
        with tf.variable_scope('input_atn') as scope:
            # e1 attention rate: softmax(e1*W*word_list)
            atn_input1_left = tf.expand_dims(tf.matmul(e1_input, weights['input_attention']), 1)
            atn_input1_rate = tf.nn.softmax(tf.matmul(atn_input1_left, tf.transpose(input_dropout,[0,2,1])))
            atn_input1_rate = tf.matrix_diag(tf.reshape(atn_input1_rate, [-1,self.max_sentence_len]))
            # e2 attention rate: softmax(e2*W*word_list)
            atn_input2_left = tf.expand_dims(tf.matmul(e2_input, weights['input_attention']), 1)
            atn_input2_rate = tf.nn.softmax(tf.matmul(atn_input2_left, tf.transpose(input_dropout,[0,2,1])))
            atn_input2_rate = tf.matrix_diag(tf.reshape(atn_input2_rate, [-1,self.max_sentence_len]))
            # 将e1、e2的attention rate拼接到一起
            atn_input_out = tf.matmul((atn_input1_rate+atn_input2_rate)/2, input_dropout)
        #-------------------------------------- convolution_layer ----------------------------------------------
        with tf.variable_scope('convolution_layer'):
            conv_hidden = tf.nn.conv1d(atn_input_out,weights['conv'] , 1, 'SAME')
            conv_out = tf.nn.relu(conv_hidden + biases['conv'])
            pool_out = tf.nn.max_pool(tf.expand_dims(conv_out,-1), ksize=[1,self.max_sentence_len,1,1], strides=[1,self.max_sentence_len,1,1], padding='SAME')
            pool_out = tf.reshape(pool_out, [-1, self.conv_size])
        #-------------------------------------- fully_connected ----------------------------------------------
        with tf.variable_scope('fully_connected'):
            full_conn_out = tf.nn.relu(tf.matmul(pool_out, weights['full_conn']) + biases['full_conn'])
        #-------------------------------------- pool attention ----------------------------------------------
#         with tf.variable_scope('pool_atn') as scope:
#             # attention rate: softmax(full_conn*W*pool_out)
#             atn_pool_left = tf.expand_dims(tf.matmul(full_conn_out, weights['pool_attention']), 1)
#             atn_pool_rate = tf.nn.softmax(tf.matmul(atn_pool_left, tf.transpose(pool_out,[0,2,1])))
#             atn_pool_out = tf.reshape(tf.matmul(atn_pool_rate, pool_out), [-1, self.pool_out_size])
        with tf.variable_scope('output'):
            out_dropout = tf.nn.dropout(full_conn_out, dropout_input[1])
            pred_label = tf.matmul(tf.nn.sigmoid(out_dropout), weights['out']) + biases['out']
        return pred_label, words_input, position_input, entity_input, dropout_input
    
    
    def loss_evaluation(self, pred_label):
        #-------------------------------------- Define loss and evaluation --------------------------------------
        correct_label_input = tf.placeholder('int32', [None, self.relation_classes], name='correct_label_input')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=correct_label_input))
    #     correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(label,1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return loss, correct_label_input
    
    
    def train(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
#         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        return optimizer, init
    
