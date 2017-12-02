'''
Created on 2017年4月4日

@author: zry
'''
import re
import random
from nltk import StanfordPOSTagger

if __name__ == '__main__':
    w2v_path = r'D:\myDoc\study\语料\wordVec_rnnlm\rnnlm_word.txt'
    semEval_path = r'D:\myDoc\study\语料\SemEval2010_task8\test\TEST_FILE_FULL.TXT'
    output_file = open(r'C:\Users\zry\Desktop\1.txt','w')
    word_list = []
    # 读取w2v的词表
    for line in open(w2v_path):
        word_list.append(line[:-1].lower())
    with open(semEval_path) as sem_file:
        line = sem_file.readline()
        i = 0
        while line != '':
            i += 1
            if i%500==0:
                print('-------------------'+str(i)+'-------------------')
            # 获取句子，只保留英文、数字、命名体表示符<e1></e1><e2></e2>
            sentence = line[line.find('\"')+1:-3]
            sentence_copy = sentence.lower()
            sentence = re.sub(r"[^a-zA-Z0-9</>]", " ", sentence)
            word_set = sentence.split(" ")
            word_id_seq = []
            word_str_seq = []
            entity_location = [0,0,0,0]
            # 句子中的词列表，保存w2v的编号。不存在则保存0
            empty_word_count = 0
            entity1 = ""
            entity2 = ""
            for word_index,word in enumerate(word_set):
                if word.strip()=='':
                    empty_word_count += 1
                    continue
                word = word.lower()
                # 将数字开头的单词替换为'number'单词，如:20cm -> number
                if re.match('^[0-9]', word):
                    word = 'number'
                # 记录命名体位置，index，从0开始
                if word.startswith('<e1>'):
                    word = word.replace('<e1>','')
                    word_set[word_index] = word
                    entity_location[0] = word_index-empty_word_count
                if word.startswith('<e2>'):
                    word = word.replace('<e2>','')
                    word_set[word_index] = word
                    entity_location[2] = word_index-empty_word_count
                if word.endswith('</e1>'):
                    word = word.replace('</e1>','')
                    word_set[word_index] = word
                    entity_location[1] = word_index-empty_word_count
                if word.endswith('</e2>'):
                    word = word.replace('</e2>','')
                    word_set[word_index] = word
                    entity_location[3] = word_index-empty_word_count
                word_str_seq.append(word)
                # 查询单词,保存w2v中的编号
                if word_list.__contains__(word):
                    word_id_seq.append(str(word_list.index(word)+1))
                else:
                    word_id_seq.append('0')
            output_file.write(' '.join(word_id_seq)+'\n')
            output_file.write(' '.join([str(item) for item in entity_location])+'\n')
            # 与命名体1相对位置
            relate_position_str = ""
            for j in range(len(word_id_seq)):
                if j<entity_location[0]:
                    relate_position_str += str(j-entity_location[0])+" "
                elif j>entity_location[1]:
                    relate_position_str += str(j-entity_location[1])+" "
                else:
                    relate_position_str += "0 "
            output_file.write(relate_position_str[:-1]+'\n')
            # 与命名体2相对位置
            relate_position_str = ""
            for j in range(len(word_id_seq)):
                if j<entity_location[2]:
                    relate_position_str += str(j-entity_location[2])+" "
                elif j>entity_location[3]:
                    relate_position_str += str(j-entity_location[3])+" "
                else:
                    relate_position_str += "0 "
            output_file.write(relate_position_str[:-1]+'\n')
#             output_file.write(' '.join(word_str_seq)+'\n')
            # relation type
            line = sem_file.readline()
            rel_type = line[:line.find('(')]
            output_file.write(rel_type+'\n')
            # next sentence
            line = sem_file.readline()
            line = sem_file.readline()
            line = sem_file.readline()
    output_file.close()
    