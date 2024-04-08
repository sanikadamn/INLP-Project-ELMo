from indicnlp.tokenize import indic_tokenize
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

max_sen_len = 30
max_wd_len = 10


def fix(lst, length, pad_val):
    if len(lst) < length:
        lst += [pad_val] * (length - len(lst))
        return lst
    else:
        return lst[:length]


def get_char_idx(sen_list, char_to_idx):
    char_idx = []
    for sentence in tqdm(sen_list, desc="Processing sentences"):
        sentence_idx = []
        for word in sentence:
            word_idx = []
            for char in word:
                word_idx.append(char_to_idx[char])
            sentence_idx.append(fix(word_idx, max_wd_len, 0))
        char_idx.append(fix(sentence_idx, max_sen_len, [0] * max_wd_len))
    return char_idx


def get_char_idx_flat(sen_list, char_to_idx):
    char_idx = []
    for word in tqdm(sen_list, desc="Processing words"):
        word_idx = []
        for char in word:
            word_idx.append(char_to_idx[char])
        char_idx.append(fix(word_idx, max_wd_len, 0))
    return char_idx


def get_loaders(
    path,
    num_sentences=1000000,
    train_proportion=0.8,
    dev_proportion=0.1,
):
    sentences = []

    with open(path, "r", encoding="utf-8") as f:

        for i in tqdm(range(num_sentences), desc="Reading sentences"):
            sentences.append(f.readline())

    f.close()

    sens = [indic_tokenize.trivial_tokenize(sentence) for sentence in sentences]

    punctuations = [
        "।",
        ",",
        ".",
        "?",
        "!",
        ";",
        ":",
        "-",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "‘",
        "’",
        '"',
        "'",
        f"\n",
    ]

    sens = [
        [token for token in sentence if token not in punctuations] for sentence in sens
    ]

    char_to_idx = {}
    idx_to_char = {}
    char_to_idx["<pad>"] = 0
    idx_to_char[0] = "<pad>"
    char_to_idx["<unk>"] = 1
    idx_to_char[1] = "<unk>"
    char_to_idx["<s>"] = 2
    idx_to_char[2] = "<s>"

    # add 5 <s> tokens to the beginning of each sentence
    for sentence in sens:
        sentence.insert(0, "<s>")
        sentence.insert(0, "<s>")
        sentence.insert(0, "<s>")
        sentence.insert(0, "<s>")
        sentence.insert(0, "<s>")

    # make the character to index and index to character mappings
    for sentence in tqdm(sens, desc="Mapping characters"):
        for word in sentence:
            for char in word:
                if char not in char_to_idx:
                    char_to_idx[char] = len(char_to_idx)
                    idx_to_char[len(idx_to_char)] = char

    # train, test, dev split
    train_sens = sens[: (int)(train_proportion * num_sentences)]
    dev_sens = sens[
        (int)(train_proportion * num_sentences) : (int)(
            train_proportion * num_sentences
        )
        + (int)(dev_proportion * num_sentences)
    ]
    test_sens = sens[
        (int)(train_proportion * num_sentences)
        + (int)(dev_proportion * num_sentences) :
    ]

    # get the character indices - basically get a list of lists of lists for each
    # list of sentences, each of which is a list of words, each of which is a list of characters
    train_ids = get_char_idx(train_sens, char_to_idx)
    dev_ids = get_char_idx(dev_sens, char_to_idx)
    test_ids = get_char_idx(test_sens, char_to_idx)

    class nextwordDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data  # list of lists of lists

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # this should return the input and output for the idx-th sentence
            # input is the previous 5 words, output is the next word
            # return a list of tuples of the form (input, output)

            sentence = self.data[idx]
            input = []
            output = []

            for i in range(5, len(sentence)):
                input.append(sentence[i - 5 : i])
                output.append(sentence[i])

            return torch.tensor(input), torch.tensor(output)

    # train_tensor = torch.tensor(train_ids)
    # dev_tensor = torch.tensor(dev_ids)
    # test_tensor = torch.tensor(test_ids)

    train_loader = DataLoader(nextwordDataset(train_ids), batch_size=32, shuffle=False)
    dev_loader = DataLoader(nextwordDataset(dev_ids), batch_size=32, shuffle=False)
    test_loader = DataLoader(nextwordDataset(test_ids), batch_size=32, shuffle=False)

    sens_flattened = [chars for words in sens for chars in words]

    train_sens_flat = sens_flattened[:80000]
    dev_sens_flat = sens_flattened[80000:90000]
    test_sens_flat = sens_flattened[90000:]

    train_ids_flat = get_char_idx_flat(train_sens_flat, char_to_idx)
    dev_ids_flat = get_char_idx_flat(dev_sens_flat, char_to_idx)
    test_ids_flat = get_char_idx_flat(test_sens_flat, char_to_idx)

    train_flat_tensor = torch.tensor(train_ids_flat)
    dev_flat_tensor = torch.tensor(dev_ids_flat)
    test_flat_tensor = torch.tensor(test_ids_flat)

    train_flat_loader = DataLoader(
        TensorDataset(train_flat_tensor), batch_size=32, shuffle=False
    )
    dev_flat_loader = DataLoader(
        TensorDataset(dev_flat_tensor), batch_size=32, shuffle=False
    )
    test_flat_loader = DataLoader(
        TensorDataset(test_flat_tensor), batch_size=32, shuffle=False
    )

    return (
        train_loader,
        dev_loader,
        test_loader,
        train_flat_loader,
        dev_flat_loader,
        test_flat_loader,
        char_to_idx,
        idx_to_char,
        torch.tensor(train_ids),
    )
