from indicnlp.tokenize import indic_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from preprocessing import tokenize, convert_to_oov, CharLevelVocab, WordLevelVocab
#####################################################
# Pass the character Vocab
#####################################################


torch.manual_seed(42)

OUT_OF_VOCAB = '<OOV>'
PAD_TAG = '<PAD>'
START_TAG = '<BOS>'
END_TAG = '<EOS>'

def tokenize(sentences):
    # remove punctuations
    tokenized_data = []
    for text in sentences:
        text = re.sub(r'\\', ' ', text)
        text = re.sub(r'\"', ' ', text)
        text = re.sub(r'\d+', '<NUMBER>', text)
        text = text.lower()
        text = re.sub(r'[!"#$%&\'()*–+,-./:;<=>?@^_`{|}~]', ' ', text)
        tokenized_data.append(indic_tokenize.trivial_tokenize(text))

    return tokenized_data

def split_into_characters(tokenized_data, character_vocab, word_length, sen_len):

    sentences = []
    # convert sentences inot a list of list of characters
    for sentence in tokenized_data:
        sentence_chars = []
        for word in sentence:
            word_chars = []
            for char in word:
                word_chars.append(character_vocab.char_to_index(char))
            # pad the word to a fixed length
            if len(word_chars) < word_length:
                word_chars += [0] * (word_length - len(word_chars))
            if len(word_chars) > word_length:
                word_chars = word_chars[:word_length]
            sentence_chars.append(word_chars)
        if len(sentence_chars) < sen_len:
            sentence_chars += [[0] * word_length] * (sen_len - len(sentence_chars))
        if len(sentence_chars) > sen_len:
            sentence_chars = sentence_chars[:sen_len]

        sentences.append(sentence_chars)

    return sentences

class WADataset(Dataset):
    def __init__(self, sens):
        self.sens = sens

        # print(self.sentences1[0], self.sentences2[0])
    
    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        s1 = [self.vocabulary[word] if word in self.vocabulary else self.vocabulary[OUT_OF_VOCAB] for word in self.sentences1[idx]]
        s2 = [self.vocabulary[word] if word in self.vocabulary else self.vocabulary[OUT_OF_VOCAB] for word in self.sentences2[idx]]
        return torch.tensor(s1), torch.tensor(s2), torch.tensor(self.scores[idx])

    
    def format(self, char_vocab):
        
        s1 = split_into_characters(self.sens, char_vocab, 10, 8)

        # make every word in s1 and s2 a tensor
        s1 = [[torch.tensor(word, dtype=torch.long) for word in sentence] for sentence in s1]

        # w1 is the list of every last tensor in s1
        w1 = [sentence[-1] for sentence in s1]


        return s1, w1

def pad_to_fixed_length(sequences, max_length=15, padding_value=0):
    batch_size = len(sequences)
    padded_sequences = torch.full((batch_size, max_length), padding_value, dtype=sequences[0].dtype)
    
    for i, seq in enumerate(sequences):
        # Get the length of the current sequence
        seq_length = len(seq)
        
        # Determine the number of elements to copy (either the entire sequence or up to max_length)
        copy_length = min(seq_length, max_length)
        
        # Copy the sequence into the padded tensor
        padded_sequences[i, :copy_length] = seq[:copy_length]
    
    return padded_sequences

class wo_ELMO_Dataset(Dataset):
    def __init__(self, path, word_vocab: WordLevelVocab):
        df = pd.read_csv(path, sep='\t', header=None)
        sens1 = df[5]
        sens2 = df[6]
        scores = df[4]
        self.sentences1 = tokenize(sens1)
        self.sentences2 = tokenize(sens2)
        self.scores = torch.tensor(scores)
        self.word_vocab = word_vocab
        self.max_len = 30

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        s1 = torch.tensor([self.word_vocab.word_to_index(word) for word in self.sentences1[idx]])
        s2 = torch.tensor([self.word_vocab.word_to_index(word) for word in self.sentences2[idx]])
        return torch.tensor(s1), torch.tensor(s2), torch.tensor(self.scores[idx])
    
    def format(self):
        # convert the sentences into tensors
        s1 = [torch.tensor([self.word_vocab.word_to_index(word) for word in sentence]) for sentence in self.sentences1]
        s2 = [torch.tensor([self.word_vocab.word_to_index(word) for word in sentence]) for sentence in self.sentences2]

        # pad the sentences to a fixed length of 15
        s1 = pad_to_fixed_length(s1, max_length=self.max_len, padding_value=self.word_vocab.word_to_index(PAD_TAG))
        s2 = pad_to_fixed_length(s2, max_length=self.max_len, padding_value=self.word_vocab.word_to_index(PAD_TAG))

        return s1, s2, self.scores