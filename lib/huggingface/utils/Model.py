import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
import torch
import sys

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for i,d in enumerate(data_loader):
        print("Entrenando " + "."*(i%4), end="\r")
        loss, logits = model(
            input_ids=d["input_ids"].to(device),
            attention_mask=d["attention_mask"].to(device),
            labels=F.one_hot(d['labels'].to(device), num_classes=4).float(),
            return_dict=False
        )
        logits = logits.detach().cpu().numpy()
        label_ids = d['labels'].cpu().flatten().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        targ = d['labels'].numpy()
        correct_predictions += np.sum(preds==targ)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        sys.stdout.write("\033[K")
    return correct_predictions / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    y_pred = []
    y_targ = []
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            loss, logits = model(
                input_ids = d["input_ids"].to(device),
                attention_mask = d["attention_mask"].to(device),
                labels = F.one_hot(d['labels'].to(device), num_classes=4).float(),
                return_dict=False
            )
            logits = logits.detach().cpu().numpy()
            labels_ids = d['labels'].cpu().flatten().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            targ = d['labels'].numpy()
            y_val.extend(targ.tolist())
            y_pred.extend(preds.tolist())
            correct_predictions += np.sum(preds==targ)
            losses.append(loss.item())
    return correct_predictions / n_examples, np.mean(losses), f1_score(y_targ, y_pred, average='micro')

def train(EPOCHS, model, train_data_loader, optimizer, device, scheduler, history, train_len, valid_len = 0, valid_data_loader = None):
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-'*10)
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, train_len)
        print(f'Train loss: {round(train_loss, 3)} Accuracy: {round(train_acc, 3)}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        if (valid_len > 0):
            val_acc, val_loss, f_score = eval_model(model, valid_data_loader, device, valid_len)
            print(f'Validation loss {round(val_loss, 3)} Accuracy: {round(val_acc, 3)} F1 score: {round(f_score, 3)}')
            print()
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            history['f_score'].append(f_score)
    return history

def test(model, test_data_loader, device, test_len):
    test_acc, test_loss = eval_model(model, test_data_loader, device, test_len)
    print(f'Test loss: {round(test_loss, 3)} Accuracy: {round(test_acc, 3)}')