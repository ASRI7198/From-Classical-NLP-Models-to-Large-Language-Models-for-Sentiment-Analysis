import torch
from torch.utils.data import Dataset
import Analyse.Function as fn

class SentimentDataset(Dataset): 
    def __init__(self, dataframe, vocab, label2id):
        self.texts = dataframe['clean_text'].values 
        self.labels = [label2id[label] for label in dataframe['label'].values]
        self.vocab = vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded_text = fn.preprocess_and_encode(text, self.vocab)
        one_hot = fn.one_hot_encode(encoded_text, len(self.vocab) + 1)
        
        return torch.tensor(one_hot, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        