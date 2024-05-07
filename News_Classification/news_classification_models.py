'''
Code for the models used in the news classification task.
The models are:
1. NewsClassification: A model that uses ELMo embeddings to encode the input text.
2. NewsClassification_WithoutELMo: A model that uses word embeddings to encode the input text.
'''

import torch
import torch.nn as nn

class NewsClassification(nn.Module):
    def __init__(self, elmo, embedding_dim, num_classes):
        super(NewsClassification, self).__init__()
        self.elmo = elmo
        # self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=True, num_layers=1)
        # self.fc = nn.Linear(embedding_dim*2, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim//2)
        self.fc2 = nn.Linear(embedding_dim//2, num_classes)
        self.relu = nn.ReLU()
        self.lambdas = nn.Parameter(torch.randn(3))
        self.lstm = nn.LSTM(embedding_dim, embedding_dim//2, batch_first=True, bidirectional=True, num_layers=1)

        for param in self.elmo.parameters():
            param.requires_grad = False

    def forward(self, x):
        _, _, final_embeddings = self.elmo(x)
        encoding = torch.zeros_like(final_embeddings[0])
        for i in range(3):
            encoding += self.lambdas[i] * final_embeddings[i]
        # output, (hidden, cell) = self.lstm(encoding)
        # take average of output
        # final_output = torch.mean(output, dim=1)
        output, (hidden, cell) = self.lstm(encoding)
        final_output = torch.mean(output, dim=1)
        output = self.fc(final_output)
        # output = self.fc(final_output)
        output = self.relu(output)
        output = self.fc2(output)
        return output
        

# input (batch_size, seq_len)
class NewsClassification_WithoutELMo(nn.Module):
    def __init__(self, embedding_dim, num_classes, word_vocab):
        super(NewsClassification_WithoutELMo, self).__init__()
        self.embedding = nn.Embedding(word_vocab.num_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.fc = nn.Linear(embedding_dim*2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        # take average of output
        final_output = torch.mean(output, dim=1)
        output = self.fc(final_output)
        output = self.relu(output)
        output = self.fc2(output)
        return output