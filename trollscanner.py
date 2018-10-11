"""
Created on Thu Oct 11 12:13:11 2018

@author: timon
"""
import json
import random

with open('dataset.json',encoding='utf-8') as f:
    data = (line.strip() for line in f) 
    data_json = "[{0}]".format(','.join(data))

data = json.loads(data_json)

data_list = [[]]

for elem in data:
    data_list.append([elem['content'], int(elem['annotation']['label'][0])])
    
random.shuffle(data_list)
    
print(data_list)

