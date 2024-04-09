from indicnlp.tokenize import indic_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from torchtext.vocab import build_vocab_from_iterator, Vocab
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(0)

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
            if len(word_chars) > word_length:
                word_chars = word_chars[:word_length]
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

def convert_to_tensors(words, targets, vocab, character_vocab):
    # convert the words and targets into tensors
    words_tensor = []
    for word in words:
        word_tensor = []
        for w in word:
            w_tensor = []
            for char in w:
                w_tensor.append(character_vocab[char])
            word_tensor.append(torch.tensor(w_tensor, dtype=torch.long))
        words_tensor.append(word_tensor)
    # convert the targets into a tensor
    target_tensor = []
    for target in targets:
        target_tensor.append(torch.tensor(vocab[target], dtype=torch.long))
    return words_tensor, target_tensor

OUT_OF_VOCAB = '<OOV>'
PAD_TAG = '<PAD>'
START_TAG = '<BOS>'
END_TAG = '<EOS>'

# create a torch dataset
class NextWordDataset(Dataset):
    def __init__(self, path, start=0, end=100000, vocabulary: Optional[Vocab] = None, character_vocab: Optional[Vocab] = None):
        self.sentences = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start and i < end:
                    self.sentences.append(line.strip())
                if i >= end:
                    break
        f.close()
        self.sentences = tokenize(self.sentences)
        if vocabulary is None:
            self.vocab = build_vocab_from_iterator(self.sentences, specials=[OUT_OF_VOCAB, PAD_TAG, START_TAG, END_TAG])
            self.vocab.set_default_index(self.vocab[OUT_OF_VOCAB])
        else:
            self.vocab = vocabulary

        # set default index for the vocabulary

        self.character_vocab = character_vocab
        # iterate over the dataset
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # what does this return
        # return the sentence and the next word
        # for self.vocab[0] it should return the out of vocab token
        return torch.tensor([self.vocab[word] for word in self.sentences[idx]], dtype=torch.long)

    
    def format(self, batch, window_size, returnvocab = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # convert the batch into a tensor
        # convert the batch into a tensor
        # pad the batch
        words = []
        targets = []
        # iterate over the batch
        for sentence in batch:
            for i in range(len(sentence) - window_size):
                words.append(sentence[i:i+window_size])
                targets.append(sentence[i+window_size])
        # convert the words and targets into tensors
        self.words, character_vocab = split_into_characters(words, 10)
        if self.character_vocab is None:
            self.character_vocab = character_vocab
        # convert self.vocab into a dictionary and self.character_vocab into a dictionary
        # if we are getting the vocabulary for the first time
        if returnvocab:
            vocab = {}
            for i, word in enumerate(self.vocab.get_itos()):
                vocab[word] = i
            character_vocab = {}
            for i, char in enumerate(self.character_vocab.keys()):
                character_vocab[char] = i
            words_tensor, targets = convert_to_tensors(self.words, targets, vocab, character_vocab)
            return words_tensor, targets, vocab, character_vocab
        else:
            words_tensor, targets = convert_to_tensors(self.words, targets, self.vocab, self.character_vocab)
            return words_tensor, targets



    
    
