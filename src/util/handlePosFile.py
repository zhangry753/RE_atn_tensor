'''
Created on 2017年5月18日

@author: zry
'''

if __name__ == '__main__':
    output_file = open(r'C:\Users\zry\Desktop\pos.txt','w')
    
    pos_file_line = open(r'D:\myDoc\study\文本处理\stanford-postagger-full-2016-10-31\sample-tagged.txt').readline()
    pos_list = [content[content.index('_')+1:] for content in pos_file_line[:-1].split()]
    data_file_lines = open(r'D:\myDoc\study\文本处理\stanford-postagger-full-2016-10-31\sample-input.txt').readlines()
    first_index = 0
    for line in data_file_lines:
        senten_length = len(line.split())
        pos_sub_list = pos_list[first_index : first_index+senten_length]
        first_index += senten_length
        output_file.write(" ".join(pos_sub_list)+"\n")
    output_file.flush()
    output_file.close()
        
        