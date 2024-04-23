import torch
import torch.nn as nn
from preprocessing import CharLevelDataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# make the character embedding and convolutional layer with max pooling


class CharCNN(nn.Module):
    def __init__(self, character_embedding_size, num_filters, kernel_size, max_word_length, char_vocab_size, word_embedding_dim, device=None):
        super(CharCNN, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.char_embedding = nn.Embedding(char_vocab_size, character_embedding_size).to(device)
        self.conv_layers = nn.ModuleList([nn.Conv1d(character_embedding_size, num_filters, kernel_size).to(device) for _ in range(max_word_length - kernel_size + 1)])
        self.fc = nn.Linear(num_filters * (max_word_length - kernel_size + 1), word_embedding_dim).to(device)
        self.device = device

    def forward(self, x):
        # x is a batch of words. Each word is a list of characters (batch_size, max_word_length)
        # first, we convert the characters to embeddings
        x = x.to(self.device)
        x = self.char_embedding(x) # (batch_size, max_word_length, character_embedding_size)
        # print(x.shape)
        x = x.permute(0, 2, 1) # (batch_size, character_embedding_size, max_word_length)

        # now we run the convolutional layers
        x = [conv(x) for conv in self.conv_layers]
        
        # now we max pool
        x = [torch.max(torch.relu(conv), dim=2)[0] for conv in x]

        # now we concatenate the results
        x = torch.cat(x, dim=1) # (batch_size, num_filters * (max_word_length - kernel_size + 1))
        
        # finally, we run the fully connected layer
        x = self.fc(x)
        return x


# ELMo part
class ELMo(nn.Module):
    def __init__(self, cnn_config, elmo_config, char_vocab_size):
        # input to this is a batch of sentences. Each sentence is a list of words. Each word is a list of characters.
        super(ELMo, self).__init__()
        # first, we convert the token to a representation using character embeddings
        self.char_cnn = CharCNN(cnn_config['character_embedding_size'], 
                                cnn_config['num_filters'], 
                                cnn_config['kernel_size'], 
                                cnn_config['max_word_length'], 
                                cnn_config['char_vocab_size'],
                                elmo_config['word_embedding_dim'],  
                                device = device).to(device)
        
        # based on the number of layers as passed in the argument, sequentially have that many layers

        self.forward_lstm = nn.LSTM(elmo_config['word_embedding_dim'], elmo_config['word_embedding_dim'], num_layers=1, batch_first=True, bidirectional=False)
        self.backward_lstm = nn.LSTM(elmo_config['word_embedding_dim'], elmo_config['word_embedding_dim'], num_layers=1, batch_first=True, bidirectional=False)
        self.forward_lstms = nn.ModuleList([nn.LSTM(elmo_config['word_embedding_dim'], elmo_config['word_embedding_dim'], num_layers=1, batch_first=True, bidirectional=False) for i in range(elmo_config['num_layers'])])
        self.backward_lstms = nn.ModuleList([nn.LSTM(elmo_config['word_embedding_dim'], elmo_config['word_embedding_dim'], num_layers=1, batch_first=True, bidirectional=False) for i in range(elmo_config['num_layers'])])
        self.num_layers = elmo_config['num_layers']
        self.fc_forward = nn.Linear(elmo_config['word_embedding_dim'], elmo_config['vocab_size'])
        self.fc_backward = nn.Linear(elmo_config['word_embedding_dim'], elmo_config['vocab_size'])
        
    def forward(self, x):
        # character cnn
        # convert x to tensor
        # x = torch.stack(x, dim=0)
        x = x.permute(1, 0, 2)
        x = [self.char_cnn(word) for word in x]
        # lstm1
        x = torch.stack(x, dim=1) 

        final_embeddings = []
        forward_output = x
        # flip once, pass through backward embedding thing
        backward_output = torch.flip(x, [1])
        # for final embeddings, flip again to make sure words align
        final_embeddings.append(torch.cat([forward_output, torch.flip(backward_output, [1])], dim=2))
        
        for i in range(self.num_layers):
            forward_output, _ = self.forward_lstms[i](forward_output)
            
            # passed through the backward lstm correctly
            backward_output, _ = self.backward_lstms[i](backward_output)
            # appending flipped so it matches
            final_embeddings.append(torch.cat([forward_output, torch.flip(backward_output, [1])], dim=2))
        
        forward_output = self.fc_forward(forward_output)
        # flip backward
        # final backward output is correct
        backward_output = torch.flip(backward_output, [1])
        backward_output = self.fc_backward(backward_output)
        return forward_output, backward_output, final_embeddings