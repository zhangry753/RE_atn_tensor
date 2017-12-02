'''
Created on 2017年4月15日

@author: zry
'''
import random
import numpy as np
from gensim.models.word2vec import Word2Vec

class Batch_reader(object):
    '''
    classdocs
    '''


    def __init__(self, word_size=50, position_size=30, pos_size=25, max_sentence_len=80):
        # data
        wordVec_type = 'senna'
        wordVec_filename = 'embeddings.txt'
        wordVec_encoding = 'utf-8'
        train_set_type = 'SemEval2010_task8'
        origin_data_lines = open(r'D:\myDoc\study\语料\\'+train_set_type+r'\train\vectors_origin\origin_'+wordVec_type+'.txt').readlines()
        test_data_lines = open(r'D:\myDoc\study\语料\\'+train_set_type+r'\test\test_file_'+wordVec_type+'.txt').readlines()
#         positive_data_lines = open(r'D:\myDoc\study\语料\\'+train_set_type+r'\train\vectors_bi_classify\positive_'+wordVec_type+'.txt').readlines()
#         negative_data_lines = open(r'D:\myDoc\study\语料\\'+train_set_type+r'\train\vectors_bi_classify\negative_'+wordVec_type+'.txt').readlines()
#         classify_data_dir = r'D:\myDoc\study\语料\\'+train_set_type+r'\train\vectors_multi_classify'
#         classify_data_lines = {'Other':[],'Cause-Effect':[],'Product-Producer':[],'Entity-Origin':[],'Instrument-Agency':[],'Component-Whole':[],'Content-Container':[],'Entity-Destination':[],'Member-Collection':[],'Message-Topic':[]}
#         classify_data_lines = {'Other':[],'Cause-Effect':[],'Instrument-Agency':[],'Product-Producer':[],'Origin-Entity':[],'Theme-Tool':[],'Part-Whole':[],'Content-Container':[]}
#         for rel_type in classify_data_lines.keys():
#             classify_data_lines[rel_type] = open(classify_data_dir+"\\"+rel_type+"_"+wordVec_type+".txt").readlines()
        # word vectors
        self.max_sen_len = max_sentence_len
        self.word_size = word_size
        word_vec_embed = [[0]*word_size]
        for line in open(r'D:\myDoc\study\语料\wordVec_'+wordVec_type+'\\'+wordVec_filename,encoding=wordVec_encoding):
            word_vec_embed.append([float(item) for item in line.split()])
        # position vectors
        self.position_size = position_size
        position_vecs = Word2Vec.load(r'D:\myDoc\study\语料\position_w2v_15d')
        # pos vectors
        self.pos_size = pos_size
        pos_vecs = Word2Vec.load(r'D:\myDoc\study\语料\pos_w2v_25d')
        # relation label map
#         self.rel_class_map = {
#               'Other':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#               'Cause-Effect(e1,e2)':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#           'Product-Producer(e1,e2)':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#              'Entity-Origin(e1,e2)':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#          'Instrument-Agency(e1,e2)':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#            'Component-Whole(e1,e2)':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
#          'Content-Container(e1,e2)':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
#         'Entity-Destination(e1,e2)':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
#          'Member-Collection(e1,e2)':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
#              'Message-Topic(e1,e2)':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
#               'Cause-Effect(e2,e1)':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
#           'Product-Producer(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
#              'Entity-Origin(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
#          'Instrument-Agency(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
#            'Component-Whole(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
#          'Content-Container(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
#         'Entity-Destination(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
#          'Member-Collection(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
#              'Message-Topic(e2,e1)':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
#          }
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
#               'Cause-Effect':[1,0,0,0,0,0,0,0,0],
#           'Product-Producer':[0,1,0,0,0,0,0,0,0],
#              'Entity-Origin':[0,0,1,0,0,0,0,0,0],
#          'Instrument-Agency':[0,0,0,1,0,0,0,0,0],
#            'Component-Whole':[0,0,0,0,1,0,0,0,0],
#          'Content-Container':[0,0,0,0,0,1,0,0,0],
#         'Entity-Destination':[0,0,0,0,0,0,1,0,0],
#          'Member-Collection':[0,0,0,0,0,0,0,1,0],
#              'Message-Topic':[0,0,0,0,0,0,0,0,1]
#          }
#         self.rel_class_map = {
#             'Other':[1,0,0,0,0,0,0,0],
#             'Cause-Effect':[0,1,0,0,0,0,0,0],
#             'Instrument-Agency':[0,0,1,0,0,0,0,0],
#             'Product-Producer':[0,0,0,1,0,0,0,0],
#             'Origin-Entity':[0,0,0,0,1,0,0,0],
#             'Theme-Tool':[0,0,0,0,0,1,0,0],
#             'Part-Whole':[0,0,0,0,0,0,1,0],
#             'Content-Container':[0,0,0,0,0,0,0,1]
#         }
        #------------------------------------------------ init datas ------------------------------------------------------
        self.train_data = self.init_data((origin_data_lines,),word_vec_embed,position_vecs,pos_vecs, 0)
        self.test_data = self.init_data((test_data_lines,),word_vec_embed,position_vecs,pos_vecs, 0)
        random.shuffle(self.train_data)
        self.batch_index = 0
          
        
    def init_data(self, data_lines_tulpe, word_vec_embed, position_vec_embed, pos_vec_embed, mod=0):
        """
        args:
            data_lines_tulpe:数据集，(数据集1,数据集2...)
            word_vec_embed:词向量列表,第一个为全0表示不存在的词
            position_vec_embed:{相对位置:向量映射}字典
            pos_vec_embed:{词性:向量映射}字典
            mod:0--标准分类; 1--二分类; 2--剔除Other
        """
        data_set = []
        for data_lines in data_lines_tulpe:
            for index in range(0,len(data_lines),6):
                # 关系类型,label
                rel_class = data_lines[index+5][:-1]
                if mod==2 and rel_class=='Other':
                    continue
                elif mod==1:
                    if rel_class=='Other':
                        label = [0,1]
                    else:
                        label = [1,0]
                elif mod==0:
                    label = self.rel_class_map[rel_class]
                # 句子词向量序列
                word_vecs = [word_vec_embed[int(item)] for item in data_lines[index][:-1].split(' ')]
                sentence_len = len(word_vecs)
                # 与命名体相对位置
                e1_position = data_lines[index+3][:-1].split()
                e2_position = data_lines[index+4][:-1].split()
                position_vecs = [position_vec_embed[e1_position[i]].tolist()+position_vec_embed[e2_position[i]].tolist() for i in range(len(e1_position))]
                # 词性
                pos_vecs = [pos_vec_embed[pos].tolist() for pos in data_lines[index+1][:-1].split()]
                # 将三个输入拼接到一起
                sentence_vecs = [word_vecs[i]+position_vecs[i] for i in range(len(word_vecs))]
                # 获取命名体位置
                entity_index = [int(item) for item in data_lines[index+2][:-1].split(' ')]
                entity_index = [entity_index[0],entity_index[2]]
                #句子补零
                if len(sentence_vecs)<self.max_sen_len:
                    zeros = np.zeros((self.max_sen_len-len(sentence_vecs), self.word_size+self.position_size)).tolist()
                    sentence_vecs.extend(zeros)
                elif len(sentence_vecs)>self.max_sen_len:
                    left_index = int(len(sentence_vecs)/2 - self.max_sen_len/2)
                    right_index = int(len(sentence_vecs)/2 + self.max_sen_len/2)
                    sentence_vecs = sentence_vecs[left_index:right_index]
                data_set.append((sentence_vecs,sentence_len,entity_index,label))
        return data_set
            
            
    def read_batch(self, batch_size):
        if self.batch_index+batch_size > len(self.train_data):
            random.shuffle(self.train_data)
            self.batch_index = 0
        data_batch = self.train_data[self.batch_index : self.batch_index+batch_size]
        self.batch_index += batch_size
        senten_vecs_batch = [data[0] for data in data_batch]
        sentence_len_batch = [data[1] for data in data_batch]
        entity_indexs_batch = [data[2] for data in data_batch]
        labels_batch = [data[3] for data in data_batch]
        return senten_vecs_batch, sentence_len_batch, entity_indexs_batch, labels_batch

    def read_batch_test(self, count=None):
        if count==None:
            data_batch = self.test_data
        else:
            data_batch = random.sample(self.test_data, count)
        senten_vecs_batch = [data[0] for data in data_batch]
        sentence_len_batch = [data[1] for data in data_batch]
        entity_indexs_batch = [data[2] for data in data_batch]
        labels_batch = [data[3] for data in data_batch]
        return senten_vecs_batch, sentence_len_batch, entity_indexs_batch, labels_batch
    
    