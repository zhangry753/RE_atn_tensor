'''
Created on 2017年4月15日

@author: zry
'''
import random
import numpy as np
from gensim.models.word2vec import Word2Vec

class Batch_reader_cnn(object):
    '''
    classdocs
    '''


    def __init__(self, word_size=50, position_size=30):
        # data
        wordVec_type = 'senna'
        self.origin_data_lines = open(r'D:\myDoc\study\语料\SemEval2010_task8\SemEval2010_task8_train\vectors_origin\origin_'+wordVec_type+'.txt').readlines()
        self.test_data_lines = open(r'D:\myDoc\study\语料\SemEval2010_task8\SemEval2010_task8_test\test_file_'+wordVec_type+'.txt').readlines()
        self.positive_data_lines = open(r'D:\myDoc\study\语料\SemEval2010_task8\SemEval2010_task8_train\vectors_bi_classify\positive_'+wordVec_type+'.txt').readlines()
        self.negative_data_lines = open(r'D:\myDoc\study\语料\SemEval2010_task8\SemEval2010_task8_train\vectors_bi_classify\negative_'+wordVec_type+'.txt').readlines()
        self.classify_data_dir = r'D:\myDoc\study\语料\SemEval2010_task8\SemEval2010_task8_train\vectors_multi_classify'
        self.classify_data_lines = {'Other':[],'Cause-Effect':[],'Product-Producer':[],'Entity-Origin':[],'Instrument-Agency':[],'Component-Whole':[],'Content-Container':[],'Entity-Destination':[],'Member-Collection':[],'Message-Topic':[]}
        for rel_type in self.classify_data_lines.keys():
            self.classify_data_lines[rel_type] = open(self.classify_data_dir+"\\"+rel_type+"_"+wordVec_type+".txt").readlines()
        # word vectors
        self.word_vec_embed = [[0]*word_size]
        for line in open(r'D:\myDoc\study\语料\wordVec_'+wordVec_type+'\embeddings.txt'):
            self.word_vec_embed.append([float(item) for item in line[:-1].split(" ")])
        # position vectors
        self.position_size = position_size
        self.position_vecs = Word2Vec.load(r'D:\myDoc\study\语料\SemEval2010_task8\position_w2v_15d')
        # relation label map
        self.rel_class_map = {
              'Other':[1,0,0,0,0,0,0,0,0,0],
              'Cause-Effect':[0,1,0,0,0,0,0,0,0,0],
          'Product-Producer':[0,0,1,0,0,0,0,0,0,0],
             'Entity-Origin':[0,0,0,1,0,0,0,0,0,0],
         'Instrument-Agency':[0,0,0,0,1,0,0,0,0,0],
           'Component-Whole':[0,0,0,0,0,1,0,0,0,0],
         'Content-Container':[0,0,0,0,0,0,1,0,0,0],
        'Entity-Destination':[0,0,0,0,0,0,0,1,0,0],
         'Member-Collection':[0,0,0,0,0,0,0,0,1,0],
             'Message-Topic':[0,0,0,0,0,0,0,0,0,1]
         }
#         self.rel_class_map = {
#             'Cause-Effect':[1,0,0,0,0,0,0,0,0],
#         'Product-Producer':[0,1,0,0,0,0,0,0,0],
#            'Entity-Origin':[0,0,1,0,0,0,0,0,0],
#        'Instrument-Agency':[0,0,0,1,0,0,0,0,0],
#          'Component-Whole':[0,0,0,0,1,0,0,0,0],
#        'Content-Container':[0,0,0,0,0,1,0,0,0],
#       'Entity-Destination':[0,0,0,0,0,0,1,0,0],
#        'Member-Collection':[0,0,0,0,0,0,0,1,0],
#            'Message-Topic':[0,0,0,0,0,0,0,0,1]
#        }
        
        
    def read_batch(self, batch_size, max_sentence_len):
        data_batch = []
        rand_lines = random.sample(range(0, len(self.origin_data_lines), 6), batch_size)
        for rand_line in rand_lines:
            # 句子词向量序列
            sentence = self.origin_data_lines[rand_line][:-1]
            word_ids = [int(item) for item in sentence.split(' ')]
            # 获取命名体位置
            entity_index = [int(item) for item in self.origin_data_lines[rand_line+2][:-1].split(' ')]
            entity_index = [entity_index[0],entity_index[2]]
            # 与命名体相对位置
            e1_position = self.origin_data_lines[rand_line+3][:-1].split()
            e2_position = self.origin_data_lines[rand_line+4][:-1].split()
            position_vecs = [self.position_vecs[e1_position[i]].tolist()+self.position_vecs[e2_position[i]].tolist() for i in range(len(e1_position))]
            #句子补零
            if len(word_ids)<max_sentence_len:
                word_ids.extend([0]*(max_sentence_len-len(word_ids)))
                zeros = np.zeros((max_sentence_len-len(position_vecs), self.position_size)).tolist()
                position_vecs.extend(zeros)
            elif len(word_ids)>max_sentence_len:
                left_index = int(len(word_ids)/2 - max_sentence_len/2)
                right_index = int(len(word_ids)/2 + max_sentence_len/2)
                word_ids = word_ids[left_index:right_index]
                position_vecs = position_vecs[left_index:right_index]
            word_vecs = [self.word_vec_embed[id] for id in word_ids]
            # 关系类型,label
            rel_class = self.origin_data_lines[rand_line+5][:-1]
#             if rel_class=='Other':
#                 continue
            label = self.rel_class_map[rel_class]
            data_batch.append((word_vecs,entity_index,position_vecs,label))
        word_vecs_batch = [data[0] for data in data_batch]
        entity_index_batch = [data[1] for data in data_batch]
        position_vecs_batch = [data[2] for data in data_batch]
        labels_batch = [data[3] for data in data_batch]
        return word_vecs_batch, entity_index_batch, position_vecs_batch, labels_batch
    
    
    def read_batch_test(self, max_sentence_len,batch_size=2717):
        data_batch = []
        rand_lines = random.sample(range(0, len(self.test_data_lines), 6), batch_size)
        for rand_line in rand_lines:
            # 句子词向量序列
            sentence = self.test_data_lines[rand_line][:-1]
            word_ids = [int(item) for item in sentence.split(' ')]
            # 获取命名体位置
            entity_index = [int(item) for item in self.test_data_lines[rand_line+2][:-1].split(' ')]
            entity_index = [entity_index[0],entity_index[2]]
            # 与命名体相对位置
            e1_position = self.test_data_lines[rand_line+3][:-1].split()
            e2_position = self.test_data_lines[rand_line+4][:-1].split()
            position_vecs = [self.position_vecs[e1_position[i]].tolist()+self.position_vecs[e2_position[i]].tolist() for i in range(len(e1_position))]
            #句子补零
            if len(word_ids)<max_sentence_len:
                word_ids.extend([0]*(max_sentence_len-len(word_ids)))
                zeros = np.zeros((max_sentence_len-len(position_vecs), self.position_size)).tolist()
                position_vecs.extend(zeros)
            elif len(word_ids)>max_sentence_len:
                left_index = int(len(word_ids)/2 - max_sentence_len/2)
                right_index = int(len(word_ids)/2 + max_sentence_len/2)
                word_ids = word_ids[left_index:right_index]
                position_vecs = position_vecs[left_index:right_index]
            word_vecs = [self.word_vec_embed[id] for id in word_ids]
            # 关系类型,label
            rel_class = self.test_data_lines[rand_line+5][:-1]
#             if rel_class=='Other':
#                 continue
            label = self.rel_class_map[rel_class]
            data_batch.append((word_vecs,entity_index,position_vecs,label))
        word_vecs_batch = [data[0] for data in data_batch]
        entity_index_batch = [data[1] for data in data_batch]
        position_vecs_batch = [data[2] for data in data_batch]
        labels_batch = [data[3] for data in data_batch]
        return word_vecs_batch, entity_index_batch, position_vecs_batch, labels_batch
    
    
    