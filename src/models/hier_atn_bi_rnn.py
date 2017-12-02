'''
Created on 2017年3月23日

@author: zry
'''
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn

class Hier_atn_bi_rnn:
    def __init__(self, data_dir=None):
        # Parameters
        self.learning_rate = 0.001
        self.batch_size = 300
        # Network Parameter
        self.max_context_len = 60
        self.word_size = 50 #词向量维度
        self.position_size = int(15*2) #与命名体相对位置的维度
        self.pos_size = 25 #词性维度
        self.input_size = int(self.word_size+self.position_size+self.pos_size)
        self.bi_rnn_hidden_1 = int(80*2)
        self.bi_rnn_hidden_2 = int(80*2)
        self.attention_hidden_1 = self.input_size
        self.attention_hidden_2 = 80
        self.relation_classes = 19
        
        
    def inference(self):
        weights = {
            'attention1_h': tf.Variable(tf.random_normal([self.bi_rnn_hidden_1, 1]), name='attention1_h_w'),
            'attention1_r': tf.Variable(tf.random_normal([self.bi_rnn_hidden_1, 1]), name='attention1_r_w'),
            'attention1_out': tf.Variable(tf.random_normal([self.bi_rnn_hidden_1, self.attention_hidden_1]), name='attention1_out_w'),
            'entity': tf.Variable(tf.random_normal([self.input_size, self.attention_hidden_1]), name='entity'),
            'attention2_h': tf.Variable(tf.random_normal([self.bi_rnn_hidden_2, 1]), name='attention2_h_w'),
            'attention2_r': tf.Variable(tf.random_normal([self.bi_rnn_hidden_2, 1]), name='attention2_r_w'),
            'attention2_out': tf.Variable(tf.random_normal([self.bi_rnn_hidden_2, self.attention_hidden_2]), name='attention2_out_w'),
            'out': tf.Variable(tf.random_normal([self.attention_hidden_2, self.relation_classes]), name='out_w')
        }
        biases = {
            'attention1': tf.Variable(tf.random_normal([self.max_context_len]), name='attention1_b'),
            'attention2': tf.Variable(tf.random_normal([5]), name='attention2_b'),
            'out': tf.Variable(tf.random_normal([self.relation_classes]), name='out_b')
        }
        #-------------------------------------- input ----------------------------------------------
        dropout_input = tf.placeholder('float', [3], name='dropout_input')
        # 语境的词向量集合,[batch,第X段语境,词数量,词向量+position+pos]
        context_vecs_input = tf.placeholder('float', [None, 3, self.max_context_len, self.input_size], name='context_vecs_input')
        real_batch_size = tf.shape(context_vecs_input)[0]
        context_list = [context_vecs_input[:,0,:,:],context_vecs_input[:,1,:,:],context_vecs_input[:,2,:,:]]
        context_flat = tf.concat(context_list, 0)
        # 语境的真实长度,[batch,第X段语境]
        context_len_input = tf.placeholder('int32', [None, 3], name='context_len_input')
        context_len_list = [context_len_input[:,0],context_len_input[:,1],context_len_input[:,2]]
        context_len_flat = tf.concat(context_len_list, 0)
        # 按entity词向量,[batch,第X个,词向量+position+pos]
        entity_input = tf.placeholder('float', [None, 2, self.input_size], 'entity_input')
        #-------------------------------------- 1st bi-rnn ----------------------------------------------
        with tf.variable_scope('1st-bi-rnn'):
            lstm_cell_fw_1 = rnn.DropoutWrapper(rnn.BasicLSTMCell(int(self.bi_rnn_hidden_1/2)), output_keep_prob=dropout_input[0])
            lstm_cell_bw_1 = rnn.DropoutWrapper(rnn.BasicLSTMCell(int(self.bi_rnn_hidden_1/2)), output_keep_prob=dropout_input[0])
            lstm_cell_mix_1 = rnn.BasicLSTMCell(self.bi_rnn_hidden_1)
            bi_rnn_outs_1,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw_1, lstm_cell_bw_1, context_flat, sequence_length=context_len_flat, dtype=tf.float32)
            bi_rnn_out_1,_ = tf.nn.dynamic_rnn(lstm_cell_mix_1, tf.concat(bi_rnn_outs_1, -1), sequence_length=context_len_flat, dtype=tf.float32)
            bi_rnn_out_last_1 = bi_rnn_out_1[:,-1,:]
        #-------------------------------------- 1st-attention-layer ----------------------------------------------
        with tf.variable_scope('1st-attention-layer'):
            # attention rate:softmax(tanh(rnn_out*W1 + rnn_Last*W2 + b))
            attention_r_1 = tf.matmul(bi_rnn_out_last_1, weights['attention1_r'])
            attention_r_1 = tf.matmul(attention_r_1, tf.zeros([1,self.max_context_len])+1) #复制多份，适应句长
            attention_h_1 = tf.reshape(tf.matmul(tf.reshape(bi_rnn_out_1,[-1,self.bi_rnn_hidden_1]), weights['attention1_h']), [-1,self.max_context_len])
            attention_rate_1 = tf.expand_dims(tf.nn.softmax(tf.nn.tanh(attention_h_1 + attention_r_1 + biases['attention1'])) ,1)
            attention_out_1 = tf.matmul(tf.reshape(tf.matmul(attention_rate_1, bi_rnn_out_1), [-1,self.bi_rnn_hidden_1]), weights['attention1_out'])
        #-------------------------------------- mix context and entity ----------------------------------------------
        with tf.variable_scope('mix-context-entity'):
            entity_hidden = tf.reshape(tf.matmul(tf.reshape(entity_input,[-1,self.input_size]), weights['entity']),[-1,2,self.attention_hidden_1])
            context_1 = attention_out_1[:real_batch_size,:]
            context_2 = attention_out_1[real_batch_size:real_batch_size*2,:]
            context_3 = attention_out_1[real_batch_size*2:,:]
            rnn_2_input = tf.concat([context_1,entity_hidden[:,0,:],context_2,entity_hidden[:,1,:],context_3], 1)
            rnn_2_input = tf.reshape(rnn_2_input, [-1,5,self.attention_hidden_1])
            rnn_2_len = tf.zeros([real_batch_size], 'int32') + 5
        #-------------------------------------- 2nd bi-rnn ----------------------------------------------
        with tf.variable_scope('2nd-bi-rnn'):
            lstm_cell_fw_2 = rnn.DropoutWrapper(rnn.BasicLSTMCell(int(self.bi_rnn_hidden_2/2)), output_keep_prob=dropout_input[1])
            lstm_cell_bw_2 = rnn.DropoutWrapper(rnn.BasicLSTMCell(int(self.bi_rnn_hidden_2/2)), output_keep_prob=dropout_input[1])
            lstm_cell_mix_2 = rnn.BasicLSTMCell(self.bi_rnn_hidden_2)
            bi_rnn_outs_2,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw_2, lstm_cell_bw_2, rnn_2_input, sequence_length=rnn_2_len, dtype=tf.float32)
            bi_rnn_out_2,_ = tf.nn.dynamic_rnn(lstm_cell_mix_2, tf.concat(bi_rnn_outs_2, -1), sequence_length=rnn_2_len, dtype=tf.float32)
            bi_rnn_out_last_2 = bi_rnn_out_2[:,-1,:]
        #-------------------------------------- 2nd-attention-layer ----------------------------------------------
        with tf.variable_scope('2nd-attention-layer'):
            # attention rate:softmax(tanh(rnn_out*W1 + rnn_Last*W2 + b))
            attention_r_2 = tf.matmul(bi_rnn_out_last_2, weights['attention2_r'])
            attention_r_2 = tf.matmul(attention_r_2, tf.zeros([1,5])+1) #复制多份，适应句长
            attention_h_2 = tf.reshape(tf.matmul(tf.reshape(bi_rnn_out_2,[-1,self.bi_rnn_hidden_2]), weights['attention2_h']), [-1,5])
            attention_rate_2 = tf.expand_dims(tf.nn.softmax(tf.nn.tanh(attention_h_2 + attention_r_2 + biases['attention2'])) ,1)
            attention_out_2 = tf.matmul(tf.reshape(tf.matmul(attention_rate_2, bi_rnn_out_2), [-1,self.bi_rnn_hidden_2]), weights['attention2_out'])
        with tf.variable_scope('output'):
            attention_out_dropout = tf.nn.dropout(attention_out_2, dropout_input[2])
            pred_label = tf.matmul(tf.nn.tanh(attention_out_dropout), weights['out']) + biases['out']
        return pred_label, context_vecs_input, context_len_input, entity_input, dropout_input
    
    
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
    
