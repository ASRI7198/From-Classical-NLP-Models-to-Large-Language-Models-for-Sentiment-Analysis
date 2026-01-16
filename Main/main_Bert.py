from transformers import AutoTokenizer
import sys
sys.path.append('C:/Users/DELL/Desktop/NLP')
import Analyse.Function as fn
import config as args
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from Main.train_Bert import Train
from Main.test_Bert import test_model
from Architecture.Transformers import TransformerClassifier
from torch import nn
from torch.utils.data import DataLoader

from DataSet.TestDataset import TextDataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

    train_dataset = TextDataset(train_df['clean_text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_dataset   = TextDataset(val_df['clean_text'].tolist(), val_df['label'].tolist(), tokenizer)
    test_dataset  = TextDataset(test_df['clean_text'].tolist(), test_df['label'].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE_BERT, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.BATCH_SIZE_BERT)
    test_loader  = DataLoader(test_dataset, batch_size=args.BATCH_SIZE_BERT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerClassifier(model_name="bert-base-uncased",output_size=args.NUM_CLASSES,dropout=args.DROPOUT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)
    criterion = nn.CrossEntropyLoss()

    history = Train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.EPOCHS
    )

    test_model(model, test_loader, criterion, device)

    torch.save(model.state_dict(), args.PATH_SAVE_Bert)