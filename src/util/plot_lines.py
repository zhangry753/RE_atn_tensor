'''
Created on 2017年11月30日

@author: zry
'''
import json
from matplotlib import pyplot as plt

def get_data_from_json(file_path):
  f = open(file_path)
  json_data = json.load(f)
  step = [x[1] for x in json_data]
  val = [x[2] for x in json_data]
  return step, val

if __name__ == '__main__':
  step, value = get_data_from_json(r"C:\Users\zry\Desktop\run_C-.-tag-F1_atn_concat.json")
  _,value2 = get_data_from_json(r"C:\Users\zry\Desktop\run_C-.-tag-F1_rntn_atn.json")
  _,value3 = get_data_from_json(r"C:\Users\zry\Desktop\run_C-.-tag-F1_rntn.json")
  _,value4 = get_data_from_json(r"C:\Users\zry\Desktop\run_C-.-tag-F1_rntn_af_atn.json")
  _,value5 = get_data_from_json(r"C:\Users\zry\Desktop\run_C-.-tag-F1_original.json")
  plt.figure()
  plt.plot(step,value,"x-",label="attention-only")
  plt.plot(step,value2,".-",label="attention-after-tensor")
  plt.plot(step,value3,"d-",label="tensor-only")
  plt.plot(step,value4,"<-",label="tensor-after-attention")
  plt.plot(step,value5,"1-",label="original")
  plt.yticks([0, 1], ['$minimum$', 'normal'])
  plt.yscale('log', linthreshy=0.01)
  plt.legend(bbox_to_anchor=(1.0, 0.4), loc=1, borderaxespad=0.)
  plt.show()