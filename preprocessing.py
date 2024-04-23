from indicnlp.tokenize import indic_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from typing import List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(13)
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator

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
        text = re.sub(r'[!"#$%&\'()*–+,/:;<=>?@^_`{|}~]', ' ', text)
        tokenized_data.append(indic_tokenize.trivial_tokenize(text))
    return tokenized_data

def convert_to_oov(sentences):
    # convert words with frequency less than 2 to <OOV>
    word_counts = Counter([word for sentence in sentences for word in sentence])
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for word in sentence:
            if word_counts[word] < 4:
                new_sentence.append(OUT_OF_VOCAB)
            else:
                new_sentence.append(word)
        new_sentences.append(new_sentence)
    return new_sentences

# get character level vocab
class CharLevelVocab:
    def __init__(self) -> None:
        self.chartoidx = {}
        self.idxtochar = {}
        self.num_chars = 0
    def build_vocab(self, sentences, extra_tokens=None):
        # append extra tokens to list of sentences if not none
        new_sentences = list(sentences)
        if extra_tokens is not None:
            for token in extra_tokens:
                new_sentences.append([token])
        all_characters = [char for sentence in new_sentences for word in sentence for char in word]
        char_counts = Counter(all_characters)
        sorted_characters = sorted(char_counts, key=char_counts.get, reverse=True)
        self.chartoidx['<PAD>'] = 0
        self.idxtochar[0] = '<PAD>'
        self.chartoidx['<OOV>'] = 1
        self.idxtochar[1] = '<OOV>'
        for i, char in enumerate(sorted_characters):
            self.chartoidx[char] = i + 2
            self.idxtochar[i + 2] = char
        self.num_chars = len(self.chartoidx)

    def char_to_index(self, char):
        if char in self.chartoidx:
            return self.chartoidx[char]
        else:
            return self.chartoidx['<OOV>']
    
    def index_to_char(self, index):
        if index in self.idxtochar:
            return self.idxtochar[index]
        else:
            return '<OOV>'
        
class WordLevelVocab:
    def __init__(self) -> None:
        self.num_words = 0
    def build_vocab(self, sentences, extra_tokens=None):
        sentences = convert_to_oov(sentences)
        self.vocab = build_vocab_from_iterator(sentences, specials=extra_tokens)
        self.num_words = len(self.vocab)
    def word_to_index(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab['<OOV>']
    def index_to_word(self, index):
        if index in self.vocab.get_itos():
            return self.vocab.get_itos()[index]
        else:
            return '<OOV>'
        
        
        
class CharLevelDataset(Dataset):
    def __init__(self, path, start=0, end=10000, character_vocab: Optional[CharLevelVocab] = None, word_vocab: Optional[WordLevelVocab] = None, max_seq_length=20, max_word_length=10):
        sentences = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= start and i < end:
                    sentences.append(line.strip())
                if i >= end:
                    break
        f.close()
        sentences = tokenize(sentences)
        self.character_vocab = character_vocab
        if character_vocab is None:
            self.character_vocab = CharLevelVocab()
            self.character_vocab.build_vocab(sentences, extra_tokens=[OUT_OF_VOCAB, PAD_TAG, START_TAG, END_TAG])
        else:
            self.character_vocab = character_vocab
        if word_vocab is None:
            self.word_vocab = WordLevelVocab()
            self.word_vocab.build_vocab(sentences, extra_tokens=[OUT_OF_VOCAB, PAD_TAG, START_TAG, END_TAG])
        else:
            self.word_vocab = word_vocab

        # go through sentences, if sentence has more than 50% of words as <OOV>, remove it
        new_sentences = []
        for sentence in sentences:
            oov_count = 0
            for word in sentence:
                if word == OUT_OF_VOCAB:
                    oov_count += 1
            if oov_count < 0.5 * len(sentence):
                new_sentences.append(sentence)

        self.sentences = new_sentences
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length       
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return [torch.tensor([self.character_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in sentence], [torch.tensor(self.word_vocab.word_to_index(word), dtype=torch.long) for word in sentence]


    def collate_fn(self, batch):
        char_encodings, word_encodings = zip(*batch)
        bos_token = []
        for c in START_TAG:
            bos_token.append(self.character_vocab.char_to_index(c))
            
        bos_token = torch.tensor(bos_token, dtype=torch.long)
        eos_token = []
        for c in END_TAG:
            eos_token.append(self.character_vocab.char_to_index(c))
        eos_token = torch.tensor(eos_token, dtype=torch.long)

        pad_token = []
        for c in PAD_TAG:
            pad_token.append(self.character_vocab.char_to_index(c))
        pad_token = torch.tensor(pad_token, dtype=torch.long)
        # Add <BOS> and <EOS> tokens to each sentences (list) without using torch.cat
        char_encodings = [[bos_token] + sentence + [eos_token] for sentence in char_encodings]

        # pad each sentence to max_seq_length (add <PAD> tokens at the end) if sentence is longer, cut it
        char_encodings = [sentence[:self.max_seq_length] + [pad_token] * (self.max_seq_length - len(sentence)) for sentence in char_encodings]  
        # for each word, pad it to max_word_length (add <PAD> tokens at the end) if word is longer, cut it
        for i in range(len(char_encodings)):
            for j in range(len(char_encodings[i])):
                char_encodings[i][j] = torch.cat([char_encodings[i][j][:self.max_word_length], torch.tensor([self.character_vocab.char_to_index(PAD_TAG)]*(self.max_word_length - len(char_encodings[i][j])), dtype=torch.long)])
        # stack tensors to create a final tensor of shape (char_encodings_size, max_seq_length, max_word_length)
        char_encodings = torch.stack([torch.stack(sentence) for sentence in char_encodings])


        word_encodings = [[self.word_vocab.word_to_index(START_TAG)] + sentence + [self.word_vocab.word_to_index(END_TAG)] for sentence in word_encodings]
        word_encodings = [sentence[:self.max_seq_length] + [self.word_vocab.word_to_index(PAD_TAG)] * (self.max_seq_length - len(sentence)) for sentence in word_encodings]
        # if less than max_seq_length, pad it
        word_encodings = torch.tensor(word_encodings, dtype=torch.long)
        return char_encodings, word_encodings
    
