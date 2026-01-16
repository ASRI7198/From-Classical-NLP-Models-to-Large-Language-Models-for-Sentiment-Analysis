import torch

def Test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for phrase, label in test_loader:
            batch_size = phrase.size(0)
            hidden = model.initHidden(batch_size=batch_size)
            # Pour LSTM : hidden, cell = model.initHidden(batch_size=batch_size)

            for word in phrase.permute(1, 0, 2):
                output, hidden = model(word, hidden)
                # Pour LSTM : output, hidden, cell = model(word, (hidden, cell))

            preds = torch.argmax(output, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    ACC = correct / total
    print(f"ACC Data : {ACC:.4f}")
