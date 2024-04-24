from indicnlp.tokenize import indic_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import re
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(13)
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
import sys
sys.path.append('..')
from preprocessing import tokenize, convert_to_oov, CharLevelVocab, WordLevelVocab

OUT_OF_VOCAB = '<OOV>'
PAD_TAG = '<PAD>'
START_TAG = '<BOS>'
END_TAG = '<EOS>'

class SentimentAnalysisDataset(Dataset):
    def __init__(self, data, labels, word_vocab: WordLevelVocab, char_vocab: CharLevelVocab, max_seq_length=20, max_word_length=10):
        self.data = tokenize(data)
        self.labels = labels
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx]
        return [torch.tensor([self.char_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in sentence], torch.tensor(self.labels[idx], dtype=torch.long)

    def collate_fn(self, batch):
        sentences, labels = zip(*batch)

        bos_token = []
        for c in START_TAG:
            bos_token.append(self.char_vocab.char_to_index(c))
        bos_token = torch.tensor(bos_token, dtype=torch.long)
        eos_token = []
        for c in END_TAG:
            eos_token.append(self.char_vocab.char_to_index(c))
        eos_token = torch.tensor(eos_token, dtype=torch.long)
        pad_token = []
        for c in PAD_TAG:
            pad_token.append(self.char_vocab.char_to_index(c))
        pad_token = torch.tensor(pad_token, dtype=torch.long)
        # Add <BOS> and <EOS> tokens to each sentences (list)
        sentences = [[bos_token] + sentence + [eos_token] for sentence in sentences] 

        sentences = [sentence[:self.max_seq_length] + [pad_token] * (self.max_seq_length - len(sentence)) for sentence in sentences]  
        
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                sentences[i][j] = torch.cat([sentences[i][j][:self.max_word_length], torch.tensor([self.char_vocab.char_to_index(PAD_TAG)]*(self.max_word_length - len(sentences[i][j])), dtype=torch.long)])
        
        sentences = torch.stack([torch.stack(sentence) for sentence in sentences])

        # convert labels to tensors
        labels = torch.stack(labels)
        return sentences, labels
    

class EmbeddingDataset(Dataset):
    def __init__(self, data, labels, word_vocab: WordLevelVocab):
        self.data = tokenize(data)
        self.labels = labels
        self.word_vocab = word_vocab
        self.max_seq_length = 20
    def __len__(self):
        return len(self.data)   
    def __getitem__(self, idx):
        return torch.tensor([self.word_vocab.word_to_index(word) for word in self.data[idx]]), torch.tensor(self.labels[idx])
    def collate_fn(self, batch):
        sentences, labels = zip(*batch)
        # add <BOS> and <EOS> tokens
        sentences = [torch.cat((torch.tensor([self.word_vocab.word_to_index('<BOS>')]), sentence, torch.tensor([self.word_vocab.word_to_index('<EOS>')]))) for sentence in sentences]
        # pad sentences
        sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=self.word_vocab.word_to_index('<PAD>'))
        return sentences, torch.tensor(labels)

        