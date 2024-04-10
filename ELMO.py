import torch
import torch.nn as nn
from preprocessing import NextWordDataset
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

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
        self.forward_lstm = nn.LSTM(elmo_config['word_embedding_dim'], int(elmo_config['word_embedding_dim']/2), 
                                    1, bidirectional = False).to(device)
        self.backward_lstm = nn.LSTM(elmo_config['word_embedding_dim'], int(elmo_config['word_embedding_dim']/2),
                                    1, bidirectional = False).to(device)
        # based on the number of layers as passed in the argument, sequentially have that many layers
        self.forward_lstms = nn.ModuleList([self.forward_lstm for _ in range(elmo_config['num_layers'])])
        self.backward_lstms = nn.ModuleList([self.backward_lstm for _ in range(elmo_config['num_layers'])])
        self.num_layers = elmo_config['num_layers']
        self.fc = nn.Linear(elmo_config['word_embedding_dim'], elmo_config['vocab_size'])
        
    def forward(self, x):
        # character cnn
        # convert x to tensor
        x = torch.stack(x, dim=0)
        x = x.permute(1, 0, 2)
        x = [self.char_cnn(word) for word in x]
        # lstm1
        x = torch.stack(x, dim=1) 
        x = x.permute(1, 0, 2) 
        
        lstm_output = x
        for i in range(self.num_layers):
            forward_lstm_output, _ = self.forward_lstms[i](lstm_output)
            backward_lstm_output, _ = self.backward_lstms[i](torch.flip(lstm_output, [1]))
            backward_lstm_output = torch.flip(backward_lstm_output, [1])
            lstm_output = torch.cat((forward_lstm_output, backward_lstm_output), dim = 2)
        
        x = torch.mean(lstm_output, dim = 1)
        x = self.fc(x)
        return x, lstm_output
     