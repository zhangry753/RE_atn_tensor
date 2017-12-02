'''
Created on 2017年4月4日

@author: zry
'''
import os

def bi_classify(data_lines, positive_out, negative_out):
    for line in range(0, len(data_lines), 6):
        rel_type = data_lines[line+5][:-1]
        if rel_type=='Other':
            negative_out.write(data_lines[line])
            negative_out.write(data_lines[line+1])
            negative_out.write(data_lines[line+2])
            negative_out.write(data_lines[line+3])
            negative_out.write(data_lines[line+4])
            negative_out.write(data_lines[line+5])
        else:
            positive_out.write(data_lines[line])
            positive_out.write(data_lines[line+1])
            positive_out.write(data_lines[line+2])
            positive_out.write(data_lines[line+3])
            positive_out.write(data_lines[line+4])
            positive_out.write(data_lines[line+5])
    
def multi_classify(data_lines, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
#     rel_class_map = {
#             'Other':[],
#             'Cause-Effect':[],
#         'Product-Producer':[],
#        'Instrument-Agency':[],
#          'Component-Whole':[],
#        'Content-Container':[],
#       'Entity-Destination':[],
#        'Member-Collection':[],
#            'Message-Topic':[]
#        }
    rel_class_map = {
        'Other':[],
        'Cause-Effect':[],
        'Instrument-Agency':[],
        'Product-Producer':[],
        'Origin-Entity':[],
        'Theme-Tool':[],
        'Part-Whole':[],
        'Content-Container':[]
    }
    for key in rel_class_map.keys():
        rel_class_map[key] = open(out_dir+"\\"+key+"_senna.txt", 'w')
    for line in range(0, len(data_lines), 6):
        rel_type = data_lines[line+5][:-1]
        rel_class_map[rel_type].write(data_lines[line])
        rel_class_map[rel_type].write(data_lines[line+1])
        rel_class_map[rel_type].write(data_lines[line+2])
        rel_class_map[rel_type].write(data_lines[line+3])
        rel_class_map[rel_type].write(data_lines[line+4])
        rel_class_map[rel_type].write(data_lines[line+5])
    for file in rel_class_map.values():
        file.close()


if __name__ == '__main__':
    out_dir = r'C:\Users\zry\Desktop\abc'
    semEval_path = r'D:\myDoc\study\语料\SemEval2007_task4\train\vectors_origin\origin_senna.txt'
    semEval_lines = open(semEval_path).readlines()
    output_file = open(r'C:\Users\zry\Desktop\1.txt','w')
    output_file2 = open(r'C:\Users\zry\Desktop\2.txt','w')
    
#     bi_classify(semEval_lines, output_file, output_file2)
    multi_classify(semEval_lines, out_dir)
    
    output_file.close()
    output_file2.close()
        
        