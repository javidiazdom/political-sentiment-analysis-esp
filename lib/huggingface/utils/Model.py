import torch.nn.functional as F
from sklearn.metrics import f1_score
from collections import defaultdict
from utils import TweetsDataset
import numpy as np
import transformers
import pandas as pd
import torch
import sys

def test(model, test_data_loader, device, test_len):
    model = model.eval()
    correct_predictions = 0
    losses = []
    with torch.no_grad():
        for d in test_data_loader:
            loss, logits = model(
                input_ids = d["input_ids"].to(device),
                attention_mask = d["attention_mask"].to(device),
                labels = F.one_hot(d["labels"].to(device), num_classes=4).float(),
                return_dict=False
            )
            logits = logits.detach().cpu().numpy()
            labels_ids = d['labels'].cpu().flatten().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            targ = d['labels'].numpy()
            correct_predictions += np.sum(preds==targ)
            losses.append(loss.item())
    return correct_predictions / test_len, np.mean(losses)


def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for i,d in enumerate(data_loader):
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
    return correct_predictions / n_examples, np.mean(losses)

def train_model(model, epochs, train_data_loader, optimizer, device, scheduler, train_len):
    for epoch in range(epochs):
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, train_len)
         

def tune_params(learning_rate, num_epochs, batch_size, model_name, filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = pd.DataFrame({})
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    df_train = pd.read_csv("../../data/procesed.csv")
    df_test = pd.read_csv("../../data/procesed_test.csv")
    for epochs in num_epochs:
        total_steps = len(df_train) * epochs
        for bs in batch_size:
            train_data_loader = TweetsDataset.create_data_loader(df_train, tokenizer, bs)
            test_data_loader = TweetsDataset.create_data_loader(df_test, tokenizer, bs)
            for lr in learning_rate:
                print(f'Training with params: {epochs} epochs, {batch_size} batch_size, {lr} learning_rate')
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                scheduler = transformers.get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps = 0,
                    num_training_steps = total_steps
                )
                train_model(model, epochs, train_data_loader, optimizer, device, scheduler, len(df_train))
                test_acc, test_loss = test(model, test_data_loader, device, len(df_test))
                results = pd.concat([results, pd.DataFrame({
                    "epochs": epochs,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "accuracy": test_acc,
                    "loss": test_loss
                }, index=[0])]).reset_index(drop=True)
    results.to_csv(filename)

                

