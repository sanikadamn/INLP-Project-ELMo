from indicnlp.tokenize import indic_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import List, Tuple, Optional
import pandas as pd


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
        tokenized_data.append(text.split())

    return tokenized_data


# function to convert sens1 and sen2 to tensors


def split_into_characters(tokenized_data, word_length):
    characters = set()
    sentences = []
    # convert sentences inot a list of list of characters
    for sentence in tokenized_data:
        sentence_chars = []
        for word in sentence:
            word_chars = []
            for char in word:
                characters.add(char)
                word_chars.append(char)
            # pad the word to a fixed length
            if len(word_chars) < word_length:
                word_chars += ['<PAD>'] * (word_length - len(word_chars))
            if(len(word_chars) >= word_length):
                word_chars = word_chars[:word_length]

            sentence_chars.append(word_chars)
        sentences.append(sentence_chars)

    return sentences


class STSDataset(Dataset):
    def __init__(self, path, vocabulary = None):
        df = pd.read_csv(path, sep='\t', header=None)
        sens1 = df[5]
        sens2 = df[6]
        scores = df[4]
        self.vocabulary = vocabulary
        self.sentences1 = tokenize(sens1)
        self.sentences2 = tokenize(sens2)
        self.scores = scores

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx):
        s1 = [self.vocabulary[word] if word in self.vocabulary else self.vocabulary[OUT_OF_VOCAB] for word in self.sentences1[idx]]
        s2 = [self.vocabulary[word] if word in self.vocabulary else self.vocabulary[OUT_OF_VOCAB] for word in self.sentences2[idx]]
        return s1, s2, self.scores[idx]

    
    def format(self, batch, character_vocabulary) -> Tuple[torch.Tensor, torch.Tensor]:
        spl1 = split_into_characters(self.sentences1, 10)
        spl2 = split_into_characters(self.sentences2, 10)

        # replace every character in spl1 and spl2 with its corresponding index in character_vocabulary
        sens1 = []
        sens2 = []
        for sentence in spl1:
            s1 = []
            for word in sentence:
                s1.append([character_vocabulary[char] if char in character_vocabulary else character_vocabulary[PAD_TAG] for char in word])
            sens1.append(s1)
        for sentence in spl2:
            s2 = []
            for word in sentence:
                s2.append([character_vocabulary[char] if char in character_vocabulary else character_vocabulary[PAD_TAG] for char in word])
            sens2.append(s2)

        # print(len(sens1[0]), len(sens2[0]), len(self.scores))
        for i in range(len(sens1)):
            if len(sens1[i]) < 15:
                sens1[i] += torch.tensor([[character_vocabulary[PAD_TAG]] * 15] * (15 - len(sens1[i])))
            if len(sens1[i]) > 15:
                sens1[i] = torch.tensor(sens1[i][:15])
            if len(sens2[i]) < 15:
                sens2[i] += torch.tensor([[character_vocabulary[PAD_TAG]] * 15] * (15 - len(sens2[i])))
            if len(sens2[i]) > 15:
                sens2[i] = torch.tensor(sens2[i][:15])

        # for sens1 convert every word that is not a tensor to a tensor
        for i in range(len(sens1)):
            for j in range(len(sens1[i])):
                if not isinstance(sens1[i][j], torch.Tensor):
                    sens1[i][j] = torch.tensor(sens1[i][j])
        # for sens2 convert every word that is not a tensor to a tensor
        for i in range(len(sens2)):
            for j in range(len(sens2[i])):
                if not isinstance(sens2[i][j], torch.Tensor):
                    sens2[i][j] = torch.tensor(sens2[i][j])
        # return torch.tensor(sens1), torch.tensor(sens2), torch.tensor(self.scores)
        return sens1, sens2, torch.tensor(self.scores)

        #convert sens1 and sens2 into tensors