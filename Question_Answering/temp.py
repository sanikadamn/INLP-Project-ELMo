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



class QuestionAnsweringDataset(Dataset):
    def __init__(self, question, answers, correct_answer, word_vocab: WordLevelVocab, char_vocab: CharLevelVocab, max_seq_length=50, max_word_length=10):
        self.question = tokenize(question)
        self.answers = answers
        self.correct_answer = correct_answer
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, idx):
        # i have 4 answers. I need to return 4 sentences with the <MASK> token replaced with the 4 different answers
        question = self.question[idx]
        answers = self.answers[idx]
        replaced = []
        for answer in answers:
            replaced_question = [word if word != 'MASK' else answer for word in question]
            replaced.append(replaced_question)
        onehot = []
        for i in range(4):
            if answers[i] == self.correct_answer[idx]:
                onehot.append(1)
            else:
                onehot.append(0)

        return [torch.tensor([self.char_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in replaced[0]], \
               [torch.tensor([self.char_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in replaced[1]], \
               [torch.tensor([self.char_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in replaced[2]], \
               [torch.tensor([self.char_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in replaced[3]], \
                torch.tensor(onehot, dtype=torch.float32)

    def collate_fn(self, batch):
        q1, q2, q3, q4, onehot = zip(*batch)
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

        q1 = [[bos_token] + sentence + [eos_token] for sentence in q1]
        q2 = [[bos_token] + sentence + [eos_token] for sentence in q2]
        q3 = [[bos_token] + sentence + [eos_token] for sentence in q3]
        q4 = [[bos_token] + sentence + [eos_token] for sentence in q4]

        q1 = [sentence[:self.max_seq_length] + [pad_token] * (self.max_seq_length - len(sentence)) for sentence in q1]
        q2 = [sentence[:self.max_seq_length] + [pad_token] * (self.max_seq_length - len(sentence)) for sentence in q2]
        q3 = [sentence[:self.max_seq_length] + [pad_token] * (self.max_seq_length - len(sentence)) for sentence in q3]
        q4 = [sentence[:self.max_seq_length] + [pad_token] * (self.max_seq_length - len(sentence)) for sentence in q4]

        for i in range(len(q1)):
            for j in range(len(q1[i])):
                q1[i][j] = torch.cat([q1[i][j][:self.max_word_length], torch.tensor([self.char_vocab.char_to_index(PAD_TAG)]*(self.max_word_length - len(q1[i][j])), dtype=torch.long)])
                q2[i][j] = torch.cat([q2[i][j][:self.max_word_length], torch.tensor([self.char_vocab.char_to_index(PAD_TAG)]*(self.max_word_length - len(q2[i][j])), dtype=torch.long)])
                q3[i][j] = torch.cat([q3[i][j][:self.max_word_length], torch.tensor([self.char_vocab.char_to_index(PAD_TAG)]*(self.max_word_length - len(q3[i][j])), dtype=torch.long)])
                q4[i][j] = torch.cat([q4[i][j][:self.max_word_length], torch.tensor([self.char_vocab.char_to_index(PAD_TAG)]*(self.max_word_length - len(q4[i][j])), dtype=torch.long)])

        q1 = torch.stack([torch.stack(sentence) for sentence in q1])
        q2 = torch.stack([torch.stack(sentence) for sentence in q2])
        q3 = torch.stack([torch.stack(sentence) for sentence in q3])
        q4 = torch.stack([torch.stack(sentence) for sentence in q4])
        
        onehot = torch.stack(onehot)
        return q1, q2, q3, q4, onehot




class QuestionAnswering(nn.Module):
    def __init__(self, elmo, embedding_dim, num_classes):
        super(QuestionAnswering, self).__init__()
        self.elmo = elmo
        self.fc = nn.Linear(embedding_dim, embedding_dim//4)
        self.fc2 = nn.Linear(embedding_dim//4, num_classes)
        self.lambdas = nn.Parameter(torch.rand(3))
        self.relu = nn.ReLU()

        for param in self.elmo.parameters():
            param.requires_grad = False

    def forward(self, q1, q2, q3, q4):
        _, _, q1 = self.elmo(q1)
        _, _, q2 = self.elmo(q2)
        _, _, q3 = self.elmo(q3)
        _, _, q4 = self.elmo(q4)
        encoding1 = torch.zeros_like(q1[0])
        encoding2 = torch.zeros_like(q2[0])
        encoding3 = torch.zeros_like(q3[0])
        encoding4 = torch.zeros_like(q4[0])
        for i in range(3):
            encoding1 += self.lambdas[i] * q1[i]
            encoding2 += self.lambdas[i] * q2[i]
            encoding3 += self.lambdas[i] * q3[i]
            encoding4 += self.lambdas[i] * q4[i]
        # take mean of the embeddings
        q1 = torch.mean(encoding1, dim=1)
        q2 = torch.mean(encoding2, dim=1)
        q3 = torch.mean(encoding3, dim=1)
        q4 = torch.mean(encoding4, dim=1)
        
        q1 = self.fc(q1)
        q2 = self.fc(q2)
        q3 = self.fc(q3)
        q4 = self.fc(q4)

        # relu
        q1 = self.relu(q1)
        q2 = self.relu(q2)
        q3 = self.relu(q3)
        q4 = self.relu(q4)

        q1 = self.fc2(q1)
        q2 = self.fc2(q2)
        q3 = self.fc2(q3)
        q4 = self.fc2(q4)
        x = torch.sum(torch.stack([q1, q2, q3, q4]), dim=0)
        x = self.relu(x)
        return x
    


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



class QuestionAnsweringDataset(Dataset):
    def __init__(self, question, answers, correct_answer, word_vocab: WordLevelVocab, char_vocab: CharLevelVocab, max_seq_length=50, max_word_length=10):
        self.question = tokenize(question)
        self.answers = answers
        self.correct_answer = correct_answer
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length

    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, idx):
        # i have 4 answers. I need to return 4 sentences with the <MASK> token replaced with the 4 different answers
        question = self.question[idx]
        answers = self.answers[idx]
        # replace mask with <> token
        question = [word if word != 'MASK' else '<>' for word in question]


        onehot = []
        for i in range(4):
            if answers[i] == self.correct_answer[idx]:
                onehot.append(1)
            else:
                onehot.append(0)


        return [torch.tensor([self.char_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in question], \
               [torch.tensor([self.char_vocab.char_to_index(char) for char in word], dtype=torch.long) for word in answers], \
               torch.tensor(onehot, dtype=torch.float32)

    def collate_fn(self, batch):
        questions, answers, onehot = zip(*batch)
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

        middle_token = []
        for c in "<>":
            middle_token.append(self.char_vocab.char_to_index(c))
        middle_token = torch.tensor(middle_token, dtype=torch.long)

        questions = [[bos_token] + sentence for sentence in questions]

        questions = [sentence[:self.max_seq_length] + [pad_token] * (self.max_seq_length - len(sentence)) for sentence in questions]

        # add the answers to the questions
        for i in range(len(questions)):
            for j in range(len(answers[i])):
                questions[i].append(middle_token)
                questions[i].append(answers[i][j])
        
        # add the end token
        for i in range(len(questions)):
            questions[i].append(eos_token)

        for i in range(len(questions)):
            for j in range(len(questions[i])):
                questions[i][j] = torch.cat([questions[i][j][:self.max_word_length], torch.tensor([self.char_vocab.char_to_index(PAD_TAG)]*(self.max_word_length - len(questions[i][j])), dtype=torch.long)])

        questions = torch.stack([torch.stack(sentence) for sentence in questions])
        
        onehot = torch.stack(onehot)
        return questions, onehot

