import torch
import torch.nn as nn
from preprocessing import CharLevelDataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('..')
from ELMO import ELMo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '/scratch/sanika/cleaned_marathi_2.txt'

num_words = 142697
num_chars = 92

# load vocab
word_vocab = torch.load('word_vocab_marathi.pt')
char_vocab = torch.load('char_vocab_marathi.pt')



def train(model, len_vocab, word_vocab, character_vocabulary):
    current = 0
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        total_loss = 0
        total_items = 0
        correct = 0

        for i in range(15):
            batch_loss = 0
            total_items_batch = 0
            correct_batch = 0
            dataset = CharLevelDataset(path=path, start=current, end=current+100000, character_vocab=character_vocabulary, word_vocab=word_vocab)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)
            current += 100000
            for batch in (dataloader):
                sentences, targets = batch
                optimizer.zero_grad()
                sentences = sentences.to(device)
                targets = targets.to(device)
                forward_output, backward_output, final_embeddings = model(sentences)
                # now, to calculate loss we need to calculate the probability of each word in the vocabulary for forward and backward separately
                # calculate the loss for the forward part
                # one hot encode the targets (shifted by 1)
                targets_forward = targets[:, 1:]
                targets_forward = torch.nn.functional.one_hot(targets_forward, num_classes=len_vocab)
                targets_forward = targets_forward.float()
                # shift the target by 1 to the other side for backward
                targets_backward = targets[:, :-1] 
                targets_backward = torch.nn.functional.one_hot(targets_backward, num_classes=len_vocab)
                targets_backward = targets_backward.float()
                # calculate the probabilities
                forward_output = forward_output[:, :-1, :]
                backward_output = backward_output[:, 1:, :]
                # calculate the loss per word
                loss = criterion(forward_output, targets_forward) + criterion(backward_output, targets_backward)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                total_loss += loss.item()
                

                # calculate accuracy, prediction is correct if it is within the top k predictions
                # return top k predictions for k = 20
                _, topk = torch.topk(forward_output, 20)
                targets_forward = torch.argmax(targets_forward, dim=2)
                targets_forward = targets_forward.unsqueeze(2)
                total_items_batch += targets_forward.shape[0]*targets_forward.shape[1]
                # check if target is in top k
                correct_predictions = targets_forward == topk
                # correct += targets_forward == topk
                correct_batch += torch.sum(correct_predictions)
                correct += torch.sum(correct_predictions)
                total_items += targets_forward.shape[0]*targets_forward.shape[1]
                
                _, topk = torch.topk(backward_output, 20)
                targets_backward = torch.argmax(targets_backward, dim=2)
                targets_backward = targets_backward.unsqueeze(2)
                total_items_batch += targets_backward.shape[0]*targets_backward.shape[1]
                # check if target is in top k
                correct_predictions = targets_backward == topk
                # correct += targets_backward == topk
                correct_batch += torch.sum(correct_predictions)
                correct += torch.sum(correct_predictions)
                total_items += targets_backward.shape[0]*targets_forward.shape[1]

            print(f'Batch Loss {i + 1}: {batch_loss}')
            print(f'Batch Accuracy {i}: {correct_batch/total_items_batch}')

        # save model
        torch.save(model.state_dict(), f'elmo_epoch_{epoch + 1}.pt')
        print(f'Epoch {epoch} Loss: {total_loss/15} Accuracy: {correct/total_items}')



if __name__ == '__main__':
    print(device)
    model = ELMo(cnn_config = {'character_embedding_size': 16, 
                           'num_filters': 32, 
                           'kernel_size': 5, 
                           'max_word_length': 10, 
                           'char_vocab_size': num_chars}, 
             elmo_config = {'num_layers': 3,
                            'word_embedding_dim': 150,
                            'vocab_size': num_words}, 
             char_vocab_size = num_chars).to(device)

    train(model, num_words, word_vocab, char_vocab)
