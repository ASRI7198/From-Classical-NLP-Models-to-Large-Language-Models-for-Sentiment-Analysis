import torch
import Analyse.Function as fn

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += fn.compute_accuracy(logits, labels)

    return total_loss / len(dataloader), total_acc / len(dataloader)



def test_model(model, test_loader, criterion, device):
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f} | Test ACC: {test_acc:.4f}")
    return test_loss, test_acc