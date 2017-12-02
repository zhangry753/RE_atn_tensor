'''
Created on 2017年3月26日

@author: zry
'''
import os    
    
from nltk.parse import stanford    
from nltk.tree import Tree

def depth_travel(root):
    result_str = '('
    for child in root:
        cur_str = ''
        if isinstance(child, Tree):
            cur_str = depth_travel(child)
            cur_str = child.label()+cur_str
        else:
            cur_str = '\"'+child+'\"'
        result_str += cur_str+"+"
    result_str = result_str[:-1]+")"
    return result_str


if __name__ == '__main__':
    #添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。    
    parser_path = r'D:\Workspace\Eclipse\Stanford-parser\lib\stanford-parser.jar'
    parser_model_path = r'D:\Workspace\Eclipse\Stanford-parser\lib\stanford-parser-2016.10.31-models.jar'
    parser_lex_model_path = r"D:\Workspace\Eclipse\Stanford-parser\model_en\lexparser\englishPCFG.ser.gz"
    #句法标注    
#     parser = stanford.StanfordParser(model_path=parser_lex_model_path, path_to_jar=parser_path, path_to_models_jar=parser_model_path)    
    parser = stanford.StanfordParser(model_path=parser_lex_model_path, path_to_jar=parser_path, path_to_models_jar=parser_model_path)    
    tree = Tree('ROOT',list(next(parser.raw_parse("the quick brown fox jumps over the lazy dog"))))
    print(depth_travel(tree))
    tree.draw()

