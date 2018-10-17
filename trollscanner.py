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
import re

class Net(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,10))     # convLayer overlaps 3 words
        self.conv4 = nn.Conv2d(1, 2, (4,10))    # convLayer overlaps 4 words
        self.conv5 = nn.Conv2d(1, 2, (5,10))    # convLayer overlaps 5 words
        self.fcl = nn.Linear(6,2)   # fully-connected-layer for the classification at the end
        

    def forward(self, input):
        embids = self.embeddings(input)
        embids = embids.unsqueeze(0)    # add two dimensions to be able to feed in the convLayer
        embids = embids.unsqueeze(0)
        
        feature3 = self.conv3(embids)   # extract the fearures with different kernels
        feature4 = self.conv4(embids)
        feature5 = self.conv5(embids)
        #print(feature3.size())
        
        feature_vec = []    # constant feature vector with size 6x1 to feed it in fcl
        feature_vec.append(torch.max(feature3[0,0,:,:]))    # get all highest values of each features of the 
        feature_vec.append(torch.max(feature3[0,1,:,:]))    # different kernels
        feature_vec.append(torch.max(feature4[0,0,:,:]))
        feature_vec.append(torch.max(feature4[0,1,:,:]))
        feature_vec.append(torch.max(feature5[0,0,:,:]))
        feature_vec.append(torch.max(feature5[0,1,:,:]))
        feature_vec = torch.Tensor(feature_vec)
        
        #print(feature_vec)
        
        output = self.fcl(feature_vec)  # pass into fcl
        output = torch.sigmoid(output)  # sigmoid is better then softmax because output is 0-1 and sums up to 1
                                        # general better for binary classification
        
        return output
        
        
        
        '''
        TODO:
        - Sometimes the sentence has only 2 words then the kernel size is greater than
          the embedings matrix
        '''
        

def getData():
    #open file and convert to a correct json
    with open('dataset.json',encoding='utf-8') as f:
        data = (line.strip() for line in f) 
        data_json = "[{0}]".format(','.join(data))
    
    data = json.loads(data_json)
    
    data_list = [[]]
    vocab = []
    
    '''
    TODO:
        seperate the words from the signs (improvable)
    '''
    
    for elem in data:
        if len(re.split('(\W+)', elem['content'])) < 5:
            continue
        tmp = int(elem['annotation']['label'][0])
        if tmp == 0:
            data_list.append([elem['content'], [0.999, 0.001]]) # [1,0] for no Troll
        else:
            data_list.append([elem['content'], [0.001, 0.999]]) # [0,1] for Troll
        vocab.extend(re.split('(\W+)', elem['content']))
     
    vocab = set(vocab)  # all words given in train_data
    print(vocab)
    vocab_size = len(vocab)
    random.shuffle(data_list)   # to get better resolds
    test_data = data_list[:1000]    # extract 1000 for testing
    train_data = data_list[1001:]
    
    return train_data, test_data, vocab, vocab_size

train_data, test_data, vocab, vocab_size= getData()

word_to_ix = {word: i for i, word in enumerate(vocab)}  # register which embedding is for which word

model = Net(vocab_size, 10)     # 10 embeddings for each word

for epoch in range(1):
    total_loss = 0
    
    for data in train_data:

        sentence = data[0]  # tweeat in string
        sentence_split = re.split('(\W+)', sentence)
        target = torch.Tensor(data[1])    # label
        sentence_idxs = torch.tensor([word_to_ix[w] for w in sentence_split], dtype=torch.long)
        
        criterion = nn.MSELoss()
        
        out = model(sentence_idxs)
        #print("out", out)
        loss = criterion(out, target)
        print("loss: ", loss)
        
        model.zero_grad()
        loss.backward()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer.step()
        
'''
TODO: Testing fucntion
'''
            
       
    '''
    sentence = train_data[0][0]
    target = train_data[0][1]
    print(sentence)
    sentence_idxs = torch.tensor([word_to_ix[w] for w in sentence.split()], dtype=torch.long)
    model(sentence_idxs)
    '''


