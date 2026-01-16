import sys
sys.path.append('C:/Users/DELL/Desktop/NLP')

import Analyse.Function as fn
import config as args
from DataSet.SentimentDataset import SentimentDataset
from torch.utils.data import DataLoader
from Architecture.RNN import RNNClassifier
from Architecture.RNN import RNNClassifier
import torch
from Main.train import Train
from Main.test import Test
from torch import nn



label2id = {
    "sadness": 0,
    "joy": 1,
    "anger": 2,
    "fear": 3,
    "love": 4,
    "surprise": 5
}



if __name__ == "__main__":

    torch.cuda.empty_cache()

    train_df = fn.load_data(args.PATH_DATA_TRAIN)
    test_df = fn.load_data(args.PATH_DATA_TEST)
    val_df = fn.load_data(args.PATH_DATA_VAL)

    train_df['clean_text'] = train_df['text'].apply(fn.clean_text)
    test_df['clean_text'] = test_df['text'].apply(fn.clean_text)
    val_df['clean_text'] = val_df['text'].apply(fn.clean_text)

    vocabulaire = fn.vocab(train_df['clean_text'].tolist())
    
    train_dataset = SentimentDataset(train_df, vocabulaire, label2id)
    val_dataset   = SentimentDataset(val_df, vocabulaire, label2id)
    test_dataset  = SentimentDataset(test_df, vocabulaire, label2id)

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)



    model = RNNClassifier(input_size=len(vocabulaire)+1, emb_size=args.EMBEDDING_DIM, hidden_size=args.HIDDEN_SIZE, output_size=args.NUM_CLASSES)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = Train(train_loader, val_loader, model)
    Test(model, test_loader)
    torch.save(model.state_dict(), args.PATH_SAVE_RNN)


