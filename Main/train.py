
import torch
import config as args

def Train(train_loader, val_loader, model):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    t = 0
    
    for epoch in range(args.EPOCHS):
        loss_total_train = 0
        for phrase, label in train_loader:
            hidden = model.initHidden(batch_size=args.BATCH_SIZE)
            # Pour LSTM : hidden, cell = model.initHidden(batch_size=args.BATCH_SIZE)
            optimizer.zero_grad()
            for word in phrase.permute(1, 0, 2):
                output_train, hidden = model(word, hidden)
                # Pour LSTM : output_train, hidden, cell = model(word, (hidden, cell))

            loss = criterion(output_train, label.to(torch.int64))
            loss_total_train += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = loss_total_train / len(train_loader)

        model.eval()
        total_loss_valid = 0
        with torch.no_grad():
            for phrase, label in val_loader:
                hidden_val = model.initHidden(batch_size=args.BATCH_SIZE)
                # Pour LSTM : hidden_val, cell_val = model.initHidden(batch_size=args.BATCH_SIZE)
                for word in phrase.permute(1, 0, 2):  
                    output_val, hidden_val = model(word, hidden_val)
                    # Pour LSTM : output_val, hidden_val, cell_val = model(word, (hidden_val, cell_val))
                loss_val = criterion(output_val, label.to(torch.int64))
                total_loss_valid += loss_val.item()

        val_loss = total_loss_valid / len(val_loader)
        print(f"Epoch {epoch}, Train loss: {train_loss} , Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            t = 0
        else:
            t += 1

        if t >= args.P:
            print("Early stopping!")
            break

    return model