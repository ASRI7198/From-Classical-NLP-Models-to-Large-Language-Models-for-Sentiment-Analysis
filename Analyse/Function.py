import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import torchtext
import re
import torch
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import torch.nn.functional as F


def load_data(path):
    data = pd.read_csv(path, sep=';', header=None, names=['text', 'label'])
    return data

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_and_encode(text, vocab):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    encoded = vocab(tokens)
    if len(encoded) < 20:
        encoded += [vocab["<pad>"]] * (20 - len(encoded))
    else:
        encoded = encoded[:20]

    return encoded

def one_hot_encode(indices, vocab_size):
    one_hot = torch.zeros(len(indices), vocab_size)
    for i, idx in enumerate(indices):
        one_hot[i, idx] = 1.0
    return one_hot



def vocab(sentences: list) -> torchtext.vocab.Vocab:
    tokenized_sentence = []
    for sentence in sentences:
        words = sentence.split()
        tokenized_sentence.append(words)
    pad_token = "<pad>"
    unk_token = "<unk>"
    vocab = build_vocab_from_iterator(tokenized_sentence, specials=[pad_token, unk_token])
    vocab.set_default_index(vocab[unk_token])
    return vocab


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


