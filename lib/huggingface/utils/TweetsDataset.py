from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

class TweetsDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer):
        self.tweets = tweets; 
        self. labels = labels;
        self.tokenizer = tokenizer
        self.max_len = 512
    def __len__(self):
        return len(self.tweets)
    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens = True,
            max_length=self.max_len,
            truncation = True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'tweet': tweet, 
            'input_ids': encoding['input_ids'].long().flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def split_test_val(df, valid_size = 0.15):
    X_train, X_val, y_train, y_val = train_test_split(df[['tweets']], df['labels'],stratify=df['labels'], test_size=valid_size, random_state = 0)
    df_train = pd.concat([pd.DataFrame({'tweets': X_train['tweets'].values}),pd.DataFrame({'labels': y_train.values})], axis = 1)
    df_valid = pd.concat([pd.DataFrame({'tweets': X_val['tweets'].values}),pd.DataFrame({'labels': y_val.values})], axis = 1)
    return df_train, df_valid


def create_data_loader(df, tokenizer, batch_size = 16):
    ds = TweetsDataset(
        tweets=df.tweets.to_numpy(),
        labels=df.labels.to_numpy(),
        tokenizer=tokenizer
    )
    return DataLoader(
        ds,
        batch_size=batch_size
    )