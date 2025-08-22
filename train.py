import json 
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import chatbotmodel

with open('D:\VS Code\html css\python\chatbot\intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))   # tuple is passed as only a single value can be appended 


ignore_words = ['?','!','.',',']

all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)


x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

BATCH_SIZE = 8


dataset = ChatDataSet()
train_loader = DataLoader( dataset = dataset,
                           batch_size = BATCH_SIZE,
                           shuffle = True,
                           num_workers = 0 )


input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
epochs = 1000


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = chatbotmodel(input_size, hidden_size, output_size).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), 
                             lr = 0.001)




for epoch in range(epochs):
    for (words, labels) in train_loader:
        model.train()
        words = words.to(device)
        labels = labels.to(device, dtype = torch.int64)
        
        preds = model(words)
        
        loss = loss_fn(preds, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch: {epoch}/{epochs} | loss: {loss}")

print(f"Final loss: {loss:.4f}")



data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "data.pth"
torch.save(data,FILE)
print(f"training completed, file save to {FILE}")


