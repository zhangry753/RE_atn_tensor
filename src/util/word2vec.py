'''
Created on 2017年5月9日

@author: zry
'''
import os
from gensim.models.word2vec import Word2Vec

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line[:-1].split()



if __name__ == '__main__':
    model = Word2Vec(MySentences(r'C:\Users\zry\Desktop\afa'), size=25, min_count=0)
    print(model['NN'])
    model.save(r'C:\Users\zry\Desktop\model')