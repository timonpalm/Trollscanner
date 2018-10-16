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
        print(feature3.size())
        
        feature_vec = []    # constant feature vector with size 6x1 to feed it in fcl
        feature_vec.append(torch.max(feature3[0,0,:,:]))    # get all highest values of each features of the 
        feature_vec.append(torch.max(feature3[0,1,:,:]))    # different kernels
        feature_vec.append(torch.max(feature4[0,0,:,:]))
        feature_vec.append(torch.max(feature4[0,1,:,:]))
        feature_vec.append(torch.max(feature5[0,0,:,:]))
        feature_vec.append(torch.max(feature5[0,1,:,:]))
        feature_vec = torch.Tensor(feature_vec)
        
        print(feature_vec)
        
        output = self.fcl(feature_vec)  # pass into fcl
        output = torch.sigmoid(output)  # sigmoid is better then softmax because output is 0-1 and sums up to 1
                                        # general better for binary classification
        
        print("output: ", output)
        
        
        
        '''
        TODO:
        Sometimes the sentence has only 2 words then the kernel size is greater than
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
    
    for elem in data:
        data_list.append([elem['content'], int(elem['annotation']['label'][0])])
        vocab.extend(elem['content'].split())
     
    vocab = set(vocab)  # all words given in train_data
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
    


