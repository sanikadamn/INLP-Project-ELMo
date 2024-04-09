from indicnlp.tokenize import indic_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence

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
            sentence_chars.append(word_chars)
        sentences.append(sentence_chars)
    # add padding to characters
    characters.add('<PAD>')
    # make the character vocab into a dictionary
    # make the pad tag the first element
    character_vocab = {}
    character_vocab['<PAD>'] = 0
    for i, char in enumerate(characters):
        character_vocab[char] = i + 1
    return sentences, character_vocab

OUT_OF_VOCAB = '<OOV>'
PAD_TAG = '<PAD>'
START_TAG = '<BOS>'
END_TAG = '<EOS>'

# create a torch dataset
class NextWordDataset(Dataset):
    def __init__(self, path, start=0, end=100000, vocabulary: Optional[Vocab] = None):
        self.sentences = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start and i < end:
                    self.sentences.append(line.strip())
                if i >= end:
                    break
        f.close()
        self.sentences = tokenize(self.sentences)
        print("Tokenized data")
        if vocabulary is None:
            self.vocab = build_vocab_from_iterator(self.sentences)
        else:
            self.vocab = vocabulary

        self.vocab.set_default_index(self.vocab['<OOV>'])
        self.sentences, self.character_vocab = split_into_characters(self.sentences, 10)
        print("Split into characters")
        # iterate over the dataset
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.vocab.lookup_indices(self.sentences[idx])), torch.tensor(self.character_vocab[self.sentences[idx]])
    
    
