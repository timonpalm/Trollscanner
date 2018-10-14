"""
Created on Thu Oct 11 12:13:11 2018

@author: timon
"""
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv2d(embedding_dim, 64, 18)
        self.conv3 = nn.Conv2d(embedding_dim, ,)
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, input):
        embids = self.embeddings(input)
        print(embids)
        

def getData():
    #open file and convert to a correct json
    with open('dataset.json',encoding='utf-8') as f:
        data = (line.strip() for line in f) 
        data_json = "[{0}]".format(','.join(data))
    
    data = json.loads(data_json)
    
    data_list = [[]]
    vocab = []
    
    for elem in data:
        data_list.append([elem['content'], int(elem['annotation']['label'][0])])
        vocab.extend(elem['content'].split())
     
    vocab = set(vocab)
    vocab_size = len(vocab)
    random.shuffle(data_list)
    test_data = data_list[:1000]
    train_data = data_list[1001:]
    
    return train_data, test_data, vocab, vocab_size

train_data, test_data, vocab, vocab_size= getData()

word_to_ix = {word: i for i, word in enumerate(vocab)}
#print(train_data)

model = Net(vocab_size, 10)

for epoch in range(1):
    total_loss = 0
    '''
    for data in train_data:
        try:
            sentence = data[0]
            target = data[1]
            sentence_idxs = torch.tensor([word_to_ix[w] for w in sentence.split()], dtype=torch.long)
            model(sentence_idxs)
            
        except: pass
    '''
    sentence = train_data[0][0]
    target = train_data[0][1]
    print(sentence)
    sentence_idxs = torch.tensor([word_to_ix[w] for w in sentence.split()], dtype=torch.long)
    model(sentence_idxs)
    


